# flake8: noqa: E402

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

# 设置numba日志级别为WARNING
logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from melo.download_utils import load_pretrain_model

# 打印环境变量以确认设置
print(f"GLOO_USE_LIBUV: {os.environ.get('GLOO_USE_LIBUV', 'Not set')}")

# 禁用 libuv
os.environ['GLOO_USE_LIBUV'] = '0'

# 启用TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encontered training problem,please try to disable TF32.
)
torch.set_float32_matmul_precision("medium")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(
#     True
# )  # Not available if torch version is lower than 2.0
torch.backends.cuda.enable_math_sdp(True)
global_step = 0


def run():
    # 获取超参数
    hps = utils.get_hparams()
    
    # 获取本地进程的rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # 如果没有初始化分布式训练，则初始化
    if not torch.distributed.is_initialized():
        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            rank=0, 
            world_size=1  # 设置为单机训练
        )
    else:
        dist.init_process_group(
            backend="gloo",
            init_method="env://",  # Due to some training problem,we proposed to use gloo instead of nccl.
            rank=local_rank,
        )  # Use torchrun instead of mp.spawn
    # 获取当前进程的rank
    rank = dist.get_rank()
    # 获取总的进程数
    n_gpus = dist.get_world_size()
    
    # 设置随机种子
    torch.manual_seed(hps.train.seed)
    # 设置当前进程的GPU设备
    torch.cuda.set_device(rank)
    # 定义全局变量
    global global_step
    # 如果当前进程的rank为0，则初始化logger和writer
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    # 打印训练数据集的路径
    print(f"hps.data.training_files={hps.data.training_files}")
    # 加载训练数据集
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    # 定义训练数据集的采样器
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # 定义数据集的合并函数
    collate_fn = TextAudioSpeakerCollate()
    # 定义训练数据集的加载器
    train_loader = DataLoader(
        train_dataset,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )  # DataLoader config could be adjusted.
    # 如果rank等于0，则加载验证集
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        # 创建验证集的DataLoader
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    # 如果模型参数中包含use_noise_scaled_mas，并且其值为True，则使用噪声缩放MAS
    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        print("Using noise scaled MAS for VITS2")
        # 初始化噪声缩放MAS
        mas_noise_scale_initial = 0.01
        # 噪声缩放MAS的增量
        noise_scale_delta = 2e-6
    # 否则，使用正常MAS
    else:
        print("Using normal MAS for VITS1")
        # 初始化噪声缩放MAS
        mas_noise_scale_initial = 0.0
        # 噪声缩放MAS的增量
        noise_scale_delta = 0.0
    # 如果模型参数中包含use_duration_discriminator，并且其值为True，则使用时长判别器
    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        # 创建时长判别器
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
    # 如果模型参数中包含use_spk_conditioned_encoder，并且其值为True，则使用说话人条件编码器
    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        # 如果说话人数量为0，则抛出异常
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    # 否则，使用正常编码器
    else:
        print("Using normal encoder for VITS1")

    # # 定义生成器网络
    net_g = SynthesizerTrn(
        len(symbols),  # 符号长度
        hps.data.filter_length // 2 + 1,  # 过滤器长度的一半加1
        hps.train.segment_size // hps.data.hop_length,  # 段大小除以跳长
        n_speakers=hps.data.n_speakers,  # 说话人数
        mas_noise_scale_initial=mas_noise_scale_initial,  # 噪声比例初始值
        noise_scale_delta=noise_scale_delta,  # 噪声比例增量
        **hps.model,  # 模型参数
    ).cuda(rank)  # 将网络移动到指定设备



    # 定义判别器网络
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)  # 将网络移动到指定设备
    # 定义生成器优化器
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),  # 过滤掉不需要梯度的参数
        hps.train.learning_rate,  # 学习率
        betas=hps.train.betas,  # beta参数
        eps=hps.train.eps,  # epsilon参数
    )
    # 定义判别器优化器
    optim_d = torch.optim.AdamW(
        net_d.parameters(),  # 判别器参数
        hps.train.learning_rate,  # 学习率
        betas=hps.train.betas,  # beta参数
        eps=hps.train.eps,  # epsilon参数
    )
    # 如果存在时长判别器，定义时长判别器优化器
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),  # 时长判别器参数
            hps.train.learning_rate,  # 学习率
            betas=hps.train.betas,  # beta参数
            eps=hps.train.eps,  # epsilon参数
        )
    else:
        optim_dur_disc = None  # 如果不存在时长判别器，优化器为None
    # 将生成器网络和判别器网络移动到指定设备
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    
    # 加载预训练模型
    pretrain_G, pretrain_D, pretrain_dur = load_pretrain_model()
    hps.pretrain_G = hps.pretrain_G or pretrain_G
    hps.pretrain_D = hps.pretrain_D or pretrain_D
    hps.pretrain_dur = hps.pretrain_dur or pretrain_dur

    # 如果存在预训练生成器模型，加载预训练模型
    if hps.pretrain_G:
        utils.load_checkpoint(
                hps.pretrain_G,
                net_g,
                None,
                skip_optimizer=True
            )
    # 如果存在预训练判别器模型，加载预训练模型
    if hps.pretrain_D:
        utils.load_checkpoint(
                hps.pretrain_D,
                net_d,
                None,
                skip_optimizer=True
            )


    # 如果存在时长判别器，加载预训练模型
    if net_dur_disc is not None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)
        if hps.pretrain_dur:
            utils.load_checkpoint(
                    hps.pretrain_dur,
                    net_dur_disc,
                    None,
                    skip_optimizer=True
                )
                
    try:
        # 如果net_dur_disc不为空，则加载最新的checkpoint
        if net_dur_disc is not None:
            # 加载最新的DUR_*.pth文件，并更新net_dur_disc和optim_dur_disc
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            # 加载最新的G_*.pth文件，并更新net_g和optim_g
            _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            # 加载最新的D_*.pth文件，并更新net_d和optim_d
            _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            # 如果optim_g的param_groups[0]中没有initial_lr，则将其设置为g_resume_lr
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            # 如果optim_d的param_groups[0]中没有initial_lr，则将其设置为d_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr
            # 如果optim_dur_disc的param_groups[0]中没有initial_lr，则将其设置为dur_resume_lr
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

        # 将epoch_str设置为最大的epoch_str和1中的最大值
        epoch_str = max(epoch_str, 1)
        # 计算global_step
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        # 打印异常信息
        print(e)
        # 将epoch_str设置为1
        epoch_str = 1
        global_step = 0

    # 定义生成器优化器的学习率调度器
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    # 定义判别器优化器的学习率调度器
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    # 如果存在时长判别器，则定义时长判别器的学习率调度器
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    # 否则，将时长判别器的学习率调度器设为None
    else:
        scheduler_dur_disc = None
    # 定义梯度缩放器
    scaler = GradScaler(enabled=hps.train.fp16_run)

    # 遍历每个epoch
    for epoch in range(epoch_str, hps.train.epochs + 1):
        # 如果当前进程是主进程
        try:
            if rank == 0:
                # 调用train_and_evaluate函数进行训练和评估
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, eval_loader],
                    logger,
                    [writer, writer_eval],
                )
            # 如果当前进程不是主进程
            else:
                # 调用train_and_evaluate函数进行训练，但不进行评估
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d, net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, None],
                    None,
                    None,
                )
        # 捕获异常
        except Exception as e:
            # 打印异常信息
            print(e)
            # 清空GPU缓存
            torch.cuda.empty_cache()
        # 更新生成器优化器的学习率
        scheduler_g.step()
        # 更新判别器优化器的学习率
        scheduler_d.step()
        # 如果存在持续时间判别器
        if net_dur_disc is not None:
            # 更新持续时间判别器的学习率
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # 设置batch_sampler的epoch
    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    # 将网络设置为训练模式
    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    # 遍历训练数据集
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        tone,
        language,
        bert,
        ja_bert,
    ) in enumerate(tqdm(train_loader)):
        # 如果使用噪声缩放，则更新噪声缩放值
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        # 将数据移动到GPU上
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        speakers = speakers.cuda(rank, non_blocking=True)
        tone = tone.cuda(rank, non_blocking=True)
        language = language.cuda(rank, non_blocking=True)
        bert = bert.cuda(rank, non_blocking=True)
        ja_bert = ja_bert.cuda(rank, non_blocking=True)

        # 开启自动混合精度训练
        with autocast(enabled=hps.train.fp16_run):
            # 前向传播
            (
                y_hat,  # 预测的音频
                l_length,  # 音频长度
                attn,  # 注意力权重
                ids_slice,  # 切片索引
                x_mask,  # 输入掩码
                z_mask,  # 隐变量掩码
                (z, z_p, m_p, logs_p, m_q, logs_q),  # 隐变量
                (hidden_x, logw, logw_),  # 隐藏状态和权重
            ) = net_g(
                x,  # 输入音频
                x_lengths,  # 输入音频长度
                spec,  # 频谱图
                spec_lengths,  # 频谱图长度
                speakers,  # 发音人
                tone,  # 音调
                language,  # 语言
                bert,  # BERT嵌入
                ja_bert,  # 日语BERT嵌入
            )
            # 将频谱图转换为梅尔频谱图
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            # 切片梅尔频谱图
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            # 将预测的音频转换为梅尔频谱图
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            # 切片音频
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            # 计算生成器生成的样本和真实样本的判别结果
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            # 禁用自动混合精度
            with autocast(enabled=False):
                # 计算判别器的损失
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                # 将判别器的损失赋值给loss_disc_all
                loss_disc_all = loss_disc
            # 如果存在持续时间判别器
            if net_dur_disc is not None:
                # 计算持续时间判别器的判别结果
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
                # 禁用自动混合精度
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    # 计算持续时间判别器的损失
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    # 将持续时间判别器的损失赋值给loss_dur_disc_all
                    loss_dur_disc_all = loss_dur_disc
                # 将持续时间判别器的优化器梯度置零
                optim_dur_disc.zero_grad()
                # 使用混合精度缩放器缩放损失
                scaler.scale(loss_dur_disc_all).backward()
                # 使用混合精度缩放器取消缩放
                scaler.unscale_(optim_dur_disc)
                # 对持续时间判别器的参数进行梯度裁剪
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                # 使用混合精度缩放器更新持续时间判别器的参数
                scaler.step(optim_dur_disc)

        # 将判别器的优化器梯度置零
        optim_d.zero_grad()
        # 使用混合精度缩放器缩放损失
        scaler.scale(loss_disc_all).backward()
        # 使用混合精度缩放器取消缩放
        scaler.unscale_(optim_d)
        # 对判别器的参数进行梯度裁剪
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        # 使用混合精度缩放器更新判别器的参数
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                # 计算时长损失
                loss_dur = torch.sum(l_length.float())
                # 计算mel损失
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # 计算kl损失
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                # 计算特征损失
                loss_fm = feature_loss(fmap_r, fmap_g)
                # 计算生成器损失
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                # 计算总损失
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    # 计算时长生成器损失
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    # 将时长生成器损失加入总损失
                    loss_gen_all += loss_dur_gen
        # 梯度清零
        optim_g.zero_grad()
        # 反向传播
        scaler.scale(loss_gen_all).backward()
        # 梯度裁剪
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        # 更新参数
        scaler.step(optim_g)
        # 更新缩放因子
        scaler.update()

        # 如果当前进程的rank为0，则执行以下操作
        if rank == 0:
            # 如果当前训练步数global_step能被log_interval整除，则执行以下操作
            if global_step % hps.train.log_interval == 0:
                # 获取当前学习率
                lr = optim_g.param_groups[0]["lr"]
                # 获取当前各个损失函数的值
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # 打印当前训练的epoch和进度
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # 打印当前各个损失函数的值、训练步数和当前学习率
                logger.info([x.item() for x in losses] + [global_step, lr])

                # 定义一个字典，用于存储当前各个损失函数的值
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                # 将当前各个损失函数的值添加到字典中
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                # 将当前各个损失函数的值添加到字典中
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                # 定义一个字典，用于存储当前各个图像的值
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                # 将当前各个图像的值和当前各个损失函数的值添加到TensorBoard中
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            # 每隔hps.train.eval_interval步数，进行一次评估
            if global_step % hps.train.eval_interval == 0:
                # 评估模型
                evaluate(hps, net_g, eval_loader, writer_eval)
                # 保存生成器模型的checkpoint
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                logger.info(f"保存生成器模型到{os.path.join(hps.model_dir, 'G_{}.pth'.format(global_step))}")
                # 保存判别器模型的checkpoint
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                # 如果存在时长判别器，则保存时长判别器的checkpoint
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )
                # 获取需要保留的checkpoint数量
                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                # 如果需要保留的checkpoint数量大于0，则清理旧的checkpoint
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        # 更新全局步数
        global_step += 1

    # 如果当前进程是主进程，则输出当前epoch
    if rank == 0:
        logger.info("====> Epoch: {}, Step: {}".format(epoch, global_step))
    # 清空cuda缓存
    torch.cuda.empty_cache()


def evaluate(hps, generator, eval_loader, writer_eval):
    # 将生成器设置为评估模式
    generator.eval()
    # 初始化图像和音频字典
    image_dict = {}
    audio_dict = {}
    # 打印评估开始
    print("Evaluating ...")
    # 不计算梯度
    with torch.no_grad():
        # 遍历评估数据加载器
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            ja_bert,
        ) in enumerate(eval_loader):
            # 将输入数据移动到GPU
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            # 遍历使用SDP的两种情况
            for use_sdp in [True, False]:
                # 使用生成器进行推理
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                # 计算预测的音频长度
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                # 将谱图转换为梅尔谱图
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                # 将预测的音频转换为梅尔谱图
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                # 将梅尔谱图添加到图像字典
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                # 将预测的音频添加到音频字典
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                # 将真实梅尔谱图添加到图像字典
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                # 将真实音频添加到音频字典
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    # 将图像和音频字典添加到TensorBoard
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    # 将生成器设置为训练模式
    generator.train()
    # 打印评估完成
    print('Evauate done')
    # 清空GPU缓存
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()

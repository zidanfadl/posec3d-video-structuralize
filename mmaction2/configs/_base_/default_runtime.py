        # default_scope (str): Used to reset registries location.
        #     Defaults to "mmengine".
default_scope = 'mmaction'


        # default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks to
        #     execute default actions like updating model parameters and saving
        #     checkpoints. Default hooks are ``OptimizerHook``,
        #     ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook`` and
        #     ``CheckpointHook``. Defaults to None.
        #     See :meth:`register_default_hooks` for more details.
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10000, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

        # custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
        #     custom actions like visualizing images processed by pipeline.
        #     Defaults to None.
# custom_hooks = [
#     dict(
#         type='EMAHook',
#         ema_type='ExpMomentumEMA',
#         momentum=0.0002,
#         update_buffers=True,
#         priority=49),
#     dict(
#         type='mmdet.PipelineSwitchHook',
#         switch_epoch=180,
#         switch_pipeline=[
#             dict(type='LoadImage', file_client_args=dict(backend='disk')),
#             dict(type='GetBBoxCenterScale'),
#             dict(
#                 type='RandomBBoxTransform',
#                 shift_factor=0.0,
#                 scale_factor=[0.75, 1.25],
#                 rotate_factor=180),
#             dict(type='RandomFlip', direction='horizontal'),
#             dict(type='TopdownAffine', input_size=(256, 256)),
#             dict(type='mmdet.YOLOXHSVRandomAug'),
#             dict(
#                 type='Albumentation',
#                 transforms=[
#                     dict(type='Blur', p=0.1),
#                     dict(type='MedianBlur', p=0.1),
#                     dict(
#                         type='CoarseDropout',
#                         max_holes=1,
#                         max_height=0.4,
#                         max_width=0.4,
#                         min_holes=1,
#                         min_height=0.2,
#                         min_width=0.2,
#                         p=0.5)
#                 ]),
#             dict(
#                 type='GenerateTarget',
#                 encoder=dict(
#                     type='SimCCLabel',
#                     input_size=(256, 256),
#                     sigma=(5.66, 5.66),
#                     simcc_split_ratio=2.0,
#                     normalize=False,
#                     use_dark=False)),
#             dict(type='PackPoseInputs')
#         ]),
#     dict(
#         type='EmptyCacheHook',
#         after_epoch=True)
# ]


        # env_cfg (dict): A dict used for setting environment. Defaults to
        #     dict(dist_cfg=dict(backend='nccl')).
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))


        # log_processor (dict, optional): A processor to format logs. Defaults to
        #     None.
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]


        # visualizer (Visualizer or dict, optional): A Visualizer object or a
        #     dict build Visualizer object. Defaults to None. If not
        #     specified, default config will be used.
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'


        # load_from (str, optional): The checkpoint file to load from.
        #     Defaults to None.
load_from = None


        # resume (bool): Whether to resume training. Defaults to False. If
        #     ``resume`` is True and ``load_from`` is None, automatically to
        #     find latest checkpoint from ``work_dir``. If not found, resuming
        #     does nothing.
resume = False

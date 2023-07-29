optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=50)
val_cfg = None
test_cfg = None


def load_model(cfg):
    model = None
    if cfg.task == "classification":
        from src.models.pose_classifier import PoseClassifier
        model = PoseClassifier(input_shape=(cfg.train.batch_size, cfg.in_channels, cfg.depth, cfg.width, cfg.height), output_dim=cfg.output_dim)
    
    return model
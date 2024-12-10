
def load_model(cfg):
    model = None
    if cfg.task == "classification":
        from src.models.pose_classifier import PoseClassifier
        model = PoseClassifier(input_shape=cfg.input_shape, output_dim=cfg.output_dim)
    
    return model
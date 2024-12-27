def load_model(cfg):
    model = None
    if cfg.task == "classification":
        from src.models.pose_classifier import PoseClassifier
        model = PoseClassifier(
            channels=cfg.in_channels, 
            output_dim=cfg.output_dim, 
            learning_rate=cfg.learning_rates.pose
        )
    if cfg.task == "privacy":
        from src.models.privacy_gan import PrivacyGAN
        model = PrivacyGAN(
            channels=cfg.in_channels, 
            output_dim_pose=cfg.output_dim,
            output_dim_privacy=(
                cfg.privacy_attributes.skin_color, 
                cfg.privacy_attributes.gender, 
                cfg.privacy_attributes.age
            ),
            loss_weights=(cfg.loss_weights.alpha, cfg.loss_weights.beta),
            learning_rates=(
                cfg.learning_rates.privatizer, 
                cfg.learning_rates.pose, 
                cfg.learning_rates.privacy
            ),
            privacy_model_type=cfg.privacy_model_type
        )
    if cfg.task == "privatizer":
        from src.models.video_privatizer import VideoPrivatizer
        model = VideoPrivatizer(
            channels=cfg.in_channels, 
            learning_rate=cfg.learning_rates.privatizer
        )
    if cfg.task == "classification_privacy":
        from src.models.privacy_classifier import PrivacyClassifier
        model = PrivacyClassifier(
            channels=cfg.in_channels, 
            output_dim=(
                cfg.privacy_attributes.skin_color, 
                cfg.privacy_attributes.gender, 
                cfg.privacy_attributes.age
            ),
            learning_rate=cfg.learning_rates.privacy
        )
    return model

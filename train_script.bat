@echo off
setlocal enabledelayedexpansion

:: Lista dei modelli di privacy
set models=VIDEO_PRIVATIZER STYLEGAN2 DEEP_PRIVACY2 BLUR PIXELATE

:: Lista degli ID dei pazienti
set ids=123 234 345

:: Valori di fps per pose_dataset
set fps_values=5 10 15 20

:: Valori di camera_type (0 per 3 canali, 1 per 1 canale)
set camera_values=0 1

:: Genera tutte le permutazioni
for %%a in (%models%) do (
    for %%b in (%ids%) do (
        for %%c in (%ids%) do (
            if not "%%b"=="%%c" (
                for %%d in (%ids%) do (
                    if not "%%d"=="%%b" if not "%%d"=="%%c" (
                        for %%e in (%fps_values%) do (
                            for %%f in (%camera_values%) do (
                                set camera_type=%%f
                                set train_ids=%%b,%%c
                                set val_test_ids=%%d
                                echo python train.py train.patient_ids="[!train_ids!]" val.patient_ids="[%%d]" test.patient_ids="[%%d]" privacy_model_type="%%a" pose_dataset.fps=%%e train.camera_type=!camera_type! val.camera_type=!camera_type! test.camera_type=!camera_type!
                                python train.py train.patient_ids="[!train_ids!]" val.patient_ids="[%%d]" test.patient_ids="[%%d]" privacy_model_type="%%a" pose_dataset.fps=%%e train.camera_type=!camera_type! val.camera_type=!camera_type! test.camera_type=!camera_type!
                            )
                        )
                    )
                )
            )
        )
    )
)

pause

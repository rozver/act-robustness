@echo off
REM -----------------------------------------------
REM Batch script to run imitate_episodes.py with thresholds 1.5,2,3 and seeds 1,2,3
REM -----------------------------------------------

REM Make sure the logs directory exists
if not exist logs (
    mkdir logs
)

REM Common parameters
set TASK_NAME=sim_transfer_cube_human
set CKPT_DIR=C:\Users\Matthew\StanfordMaterials\CS224RProject\ckpt\sim_transfer_cube_human
set POLICY_CLASS=ACT
set KL_WEIGHT=10
set CHUNK_SIZE=100
set HIDDEN_DIM=512
set BATCH_SIZE=8
set DIM_FEEDFORWARD=3200
set NUM_EPOCHS=2000
set LR=1e-5

for %%T in (1.5 2 3) do (
    for %%i in (1 2 3) do (
        echo.
        echo ================================================
        echo Running iqr_threshold=%%T, seed=%%i
        echo ================================================
        python imitate_episodes.py ^
            --task_name %TASK_NAME% ^
            --ckpt_dir "%CKPT_DIR%" ^
            --policy_class %POLICY_CLASS% ^
            --kl_weight %KL_WEIGHT% ^
            --chunk_size %CHUNK_SIZE% ^
            --hidden_dim %HIDDEN_DIM% ^
            --batch_size %BATCH_SIZE% ^
            --dim_feedforward %DIM_FEEDFORWARD% ^
            --num_epochs %NUM_EPOCHS% ^
            --lr %LR% ^
            --eval ^
            --seed %%i ^
            --temporal_agg ^
            --outlier_mbis ^
            --iqr_threshold %%T ^
            > logs\run_seed_%%T_%%i.log 2>&1
    )
)
echo.
echo All runs complete.
pause

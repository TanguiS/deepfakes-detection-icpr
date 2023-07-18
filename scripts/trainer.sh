SUBJECT_ROOT=/media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/subjects
SUBJECT_DF=/media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/subjects/dataframe.pkl

python train_binclass.py \
--net EfficientNetB4 \
--traindb subject-85-10-5 \
--valdb subject-85-10-5 \
--subject_df_path $SUBJECT_DF \
--subject_root_dir $SUBJECT_ROOT \
--face scale \
--size 256 \
--batch 64 \
--lr 1e-5 \
--valint 500 \
--patience 4 \
--maxiter 60000 \
--seed 41 \
--attention \
--device 0 \
--logint 10 \
--models_dir /media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/icpr_model/weights \
--log_dir /media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/icpr_model/graph



python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb subject-85-10-5 \
--valdb subject-85-10-5 \
--subject_df_path $SUBJECT_DF \
--subject_root_dir $SUBJECT_ROOT \
--face scale \
--size 256 \
--batch 64 \
--lr 1e-5 \
--valint 500 \
--patience 4 \
--maxiter 60000 \
--seed 41 \
--attention \
--device 0 \
--logint 10 \
--models_dir /media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/icpr_model/weights \
--log_dir /media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/icpr_model/graph

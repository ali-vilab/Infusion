inference_single_image(){
input_rgb_path=""
input_mask_path=""
input_depth_path=''
c2w=''
intri=''
output_dir=""

model_path=""
ensemble_size=1

cd ..
cd run

    CUDA_VISIBLE_DEVICES=0 python run_inference_inpainting.py \
        --input_rgb_path $input_rgb_path \
        --input_mask $input_mask_path \
        --input_depth_path $input_depth_path \
        --model_path $model_path \
        --output_dir $output_dir \
        --denoise_steps 20 \
        --intri $intri\
        --c2w $c2w \
        --use_mask \
        --blend \ 
        
        
        
}

inference_single_image
import kfp.dsl as dsl
import kfp.gcp as gcp


@dsl.pipeline(
  name='Retinal_OCT',
  description='Retinal OCT detection'
)
def dp_inf_pipe(
  # Important Parameters on top

  project_id: dsl.PipelineParam = dsl.PipelineParam(name='project-id', value="YOUR_PROJECT_ID"),
  inp_dir: dsl.PipelineParam = dsl.PipelineParam(name='input-dir', value='GCS_IMAGE_INPDIR_HERE'),
  out_dir: dsl.PipelineParam = dsl.PipelineParam(name='data-dir', value='GCS_TFRECORD_OUTDIR_HERE'),
  model_dir: dsl.PipelineParam = dsl.PipelineParam(name='model-dir', value='MODEL_CHECKPOINT_DIR_HERE'),
  save_model_dir: dsl.PipelineParam = dsl.PipelineParam(name='save-model-dir', value="DIR_TO_EXPORT_SAVED_MODEL"),
  model_name: dsl.PipelineParam = dsl.PipelineParam(name='model-name', value='MODEL_NAME_FOR_SERVING (No spaces or underscores)'),
  epochs: dsl.PipelineParam = dsl.PipelineParam(name='train-num-epochs', value=1),
  batch_size: dsl.PipelineParam = dsl.PipelineParam(name='batch-size-train', value=32),

  train_flag: dsl.PipelineParam = dsl.PipelineParam(name='train-flag', value=1),
  dataprep_flag: dsl.PipelineParam = dsl.PipelineParam(name='dataprep-flag', value=0),

  num_shards: dsl.PipelineParam = dsl.PipelineParam(name='num-shards', value=5),
  split_flag: dsl.PipelineParam = dsl.PipelineParam(name='split-flag', value=2),
  train_split: dsl.PipelineParam = dsl.PipelineParam(name='train-split', value=0.8),
  seed: dsl.PipelineParam = dsl.PipelineParam(name='seed', value=123),
  height: dsl.PipelineParam = dsl.PipelineParam(name='height', value=256),
  width: dsl.PipelineParam = dsl.PipelineParam(name='width', value=256),
  channels: dsl.PipelineParam = dsl.PipelineParam(name='channels', value=1),
  
  eval_steps: dsl.PipelineParam = dsl.PipelineParam(name='eval-steps', value=10000),
  max_train_steps: dsl.PipelineParam = dsl.PipelineParam(name='max-train-steps', value=10000),
  prefetch_buffer_size: dsl.PipelineParam = dsl.PipelineParam(name='prefetch-buffer', value=-1),

  num_gpus_serve: dsl.PipelineParam = dsl.PipelineParam(name='num-gpus-serve', value=0),
):

  dataprep = dsl.ContainerOp(
    name='dataprep',
    image='gcr.io/speedy-aurora-193605/prep_tfr_df:latest',
    arguments=["--input-dir", inp_dir,
      "--output-dir", out_dir,
      "--dataprep-flag", dataprep_flag,
      "--num-shards", num_shards,
      "--split-flag", split_flag,
      "--train-split", train_split,
      "--project-id", project_id,
      "--seed", seed,
      "--height", height,
      "--width", width,
      ],
      

      ).apply(gcp.use_gcp_secret(secret_name='user-gcp-sa', secret_file_path_in_volume='/user-gcp-sa.json', volume_name='gcp-credentials-user-gcp-sa'))

  train = dsl.ContainerOp(
    name='train',
    image='gcr.io/speedy-aurora-193605/cnn_train_dis:latest',
    arguments=["--conv-dir", out_dir,
        "--model-dir", model_dir,
        "--save-model-dir", save_model_dir,
        "--train-flag", train_flag,
        "--num-epochs", epochs,
        "--batch-size", batch_size,
        "--max-train-steps", max_train_steps,
        "--eval-steps", eval_steps,
        
        "--prefetch-buffer", prefetch_buffer_size,
        "--height", height,
        "--width", width,
        "--channels", channels,
        ]
    ).apply(gcp.use_gcp_secret(secret_name='user-gcp-sa', secret_file_path_in_volume='/user-gcp-sa.json', volume_name='gcp-credentials-user-gcp-sa'))

  tensorboard = dsl.ContainerOp(
    name='tensorboard',
    image='gcr.io/speedy-aurora-193605/model-tensorboard:latest',
    arguments=["--model-dir", model_dir,
      ],
      ).apply(gcp.use_gcp_secret(secret_name='user-gcp-sa', secret_file_path_in_volume='/user-gcp-sa.json', volume_name='gcp-credentials-user-gcp-sa'))

  tfserve = dsl.ContainerOp(
    name='tfserve',
    image='gcr.io/speedy-aurora-193605/retina-tfserve:latest',
    arguments=["--model_name", model_name,
      "--model_path", save_model_dir,
      "--num_gpus", num_gpus_serve,
      ],
      ).apply(gcp.use_gcp_secret(secret_name='admin-gcp-sa', secret_file_path_in_volume='/admin-gcp-sa.json', volume_name='gcp-credentials-admin-gcp-sa'))
      
  train.set_gpu_limit('2')
  train.set_memory_request('8G')
  train.set_cpu_request('4')
  train.after(dataprep)
  tfserve.after(train)
  tensorboard.after(dataprep)

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(dp_inf_pipe, 'retinal_oct_fin.tar.gz')
import os
import train
import pdb as bug

def kw2str(kwargs):
    args = ''
    for key, val in kwargs.items():
        if not isinstance(val, (list, tuple)):
            val = [val]
        for v in val:
            args += ' --{}={}'.format(key, str(v))
    return args


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    is_scans = True
    dataset_name = "ctscans" if is_scans else "pascal_voc_seg"
    models = ["pascal_xception", 'pascal_xception_2', 'imagenet_xception']
    model = models[0]
    output_stride = 16
    atrous_rates = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
    kwargs = dict(
        logtostderr=True,
        training_number_of_steps=30000,
        model_variant="xception_65",
        atrous_rates=atrous_rates,
        output_stride=output_stride,
        decoder_output_stride=4,
        train_crop_size=[256, 256],

        fine_tune_batch_norm=True,
        train_split="train256" if is_scans else "train",
        train_batch_size=8,
        train_logdir="logs",
        dataset=dataset_name,
        tf_initial_checkpoint="models/{}/model.ckpt".format(model),
        dataset_dir="datasets/{}/tfrecord".format(dataset_name),
        initialize_last_layer=False,
        last_layers_contain_logits_only=True,
    )
    args = kw2str(kwargs)
    print(kwargs)
    os.system("python train.py " + args)

    
if __name__ == '__main__':
    main()

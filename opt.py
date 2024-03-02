import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='Data batch size')                                                 
    parser.add_argument("--epochs", type=int, default=1,
                        help='Epoch')   
    parser.add_argument("--learning_rate", type=float, default=4e-3,
                        help='Learning rate')
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help='Warm up step')                               
    parser.add_argument("--num_training_sample", type=int, default=0,
                        help='Number of training sample')    
    parser.add_argument('--model_path', type=str, default='facebook/mbart-large-50-many-to-many-mmt',
                        help='model output dir')                                        

    parser.add_argument('--model_output_dir', type=str, default='checkpoint',
                        help='model output dir')                
    parser.add_argument('--aug_data_path', type=str, default=None,
                        help='Augmentation data path')                                  
                                                
    parser.add_argument("--train_only", type=int, default=0,
                        help='Train model')    
    parser.add_argument("--eval_only", type=int, default=0,
                        help='Eval model')                             
    parser.add_argument("--infer_only", type=int, default=0,
                        help='Infer model')                             
    parser.add_argument("--infer_data", type=str, default='My home',
                        help='Infer data')  
    parser.add_argument("--save_steps", type=int, default=5000,
                        help='Number of time steps for save model')                                                     
    parser.add_argument("--eval_steps", type=int, default=2000,
                        help='Number of time steps for eval model')                                                                             
    parser.add_argument("--lora", type=int, default=0,
                        help='Using lora')  
    parser.add_argument("--lora_rank", type=int, default=256,
                        help='Lora rank')
    parser.add_argument("--lora_alpha", type=int, default=512,
                        help='Lora alpha')
    parser.add_argument("--lora_dropout", type=int, default=0.05,
                        help='Lora dropout')
    parser.add_argument("--peft_model_id", type=str, default=None,
                        help='peft model id from hugging face')


    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()                                                
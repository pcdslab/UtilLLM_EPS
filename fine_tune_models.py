import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerForSequenceClassification, AutoModelForImageClassification,TrainingArguments,Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc,accuracy_score
import argparse
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGDatasetLLM(Dataset):
    def __init__(self, data, labels, tokenizer, max_length=2500):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        input_text = ' '.join(map(str, x.flatten()))
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        inputs['attention_mask'] = (inputs['attention_mask'].squeeze()).to(device)
        inputs['input_ids'] = (inputs['input_ids'].squeeze()).to(device)
        inputs['labels'] = (torch.tensor(label,dtype=torch.long)).to(device)
        
        return inputs

class EEGDatasetVIT(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        
        
        three_channel_data_point = np.stack([data_point*(10**6)] * 3, axis=0)
        return torch.tensor(three_channel_data_point, dtype=torch.float32), torch.tensor(self.labels[idx],dtype=torch.long)

def collate_fn(batch):
    # inputs, labels = batch
    return {
        'pixel_values': torch.stack([x[0] for x in batch]),
        'labels': torch.tensor([y[1] for y in batch])
    }
    
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    
    predictions = np.argmax(logits,axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    
    cm = confusion_matrix(labels, predictions)
    TN, FP, FN, TP = cm.ravel()
    
    acc = (TP+TN)/(TP+FP+FN+TN)
    sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    roc_auc = roc_auc_score(labels, probs)
    
    eval_dic = {"accuracy":acc,"Sensitivity":sensitivity,"Specificity":Specificity,"ROC AUC Score":roc_auc}

    return eval_dic

def Load_data(training_data_path, training_labels_path, validation_data_path, validation_labels_path):
    print('Reading data to verify correct writes ...')
    X_train_read_hdf = h5py.File(training_data_path,'r')
    X_train_read = X_train_read_hdf['tracings']
    print('Training values array shape:', X_train_read.shape)
    #X_train_read_hdf.close()
    
    y_train_read_csv = pd.read_csv(training_labels_path, header=None, index_col = None)
    y_train_read = y_train_read_csv.values.squeeze()
    print('Training labels array shape:', y_train_read.shape)       
    
    X_valid_read_hdf = h5py.File(validation_data_path,'r')
    X_valid_read = X_valid_read_hdf['tracings']      
    print('Validate values array shape:', X_valid_read.shape)
    #X_valid_read_hdf.close()  
    
    y_valid_read_csv = pd.read_csv(validation_labels_path, header=None, index_col = None)
    y_valid_read = y_valid_read_csv.values.squeeze()
    print('Validation labels array shape:', y_valid_read.shape)
    
    print('Verification Complete!')

    return X_train_read, y_train_read, X_valid_read, y_valid_read

def transform_data_point_longformer(data_point):
    transformed_data_points = []
    sub_data_points = np.array_split(data_point, 5)
    for sub_data_point in sub_data_points:
        for j in range(20):
            single_channel_data = sub_data_point[:, j]
            transformed_data_points.append(single_channel_data)
    return transformed_data_points

def transform_data_point_swinv2(data_point):
    transformed_data_points = []
    sub_data_points = np.array_split(data_point, 5)
    for sub_data_point in sub_data_points:
        replicated_data_point = np.tile(sub_data_point, (1, 12))
        last_20_columns = replicated_data_point[:, -20:]
        selected_columns_indices = np.random.choice(20, 16, replace=False)  # Randomly select 16 unique indices
        selected_columns = last_20_columns[:, selected_columns_indices]
        final_data_point = np.hstack((replicated_data_point, selected_columns))
        # replicated_data_point = np.tile(sub_data_point, (1, 13))
        # replicated_data_point = replicated_data_point[:,:256]
        transformed_data_points.append(final_data_point)
    return transformed_data_points

def transform_data_point_mit(data_point):
    transformed_data_points = []
    sub_data_points = []
    sub_data_points.append(data_point[:512])
    sub_data_points.append(data_point[512:1024])
    for sub_data_point in sub_data_points:
        replicated_data_point = np.tile(sub_data_point, (1, 25))
        last_20_columns = replicated_data_point[:, -20:]
        selected_columns_indices = np.random.choice(20, 12, replace=False)  # Randomly select 16 unique indices
        selected_columns = last_20_columns[:, selected_columns_indices]       
        final_data_point = np.hstack((replicated_data_point, selected_columns))
        transformed_data_points.append(final_data_point)
    
    return transformed_data_points

def transform_dataset(data, labels,mod):
    new_data = []
    new_labels = []
    if mod == "swinv2":
        transform_data_point = transform_data_point_swinv2
    elif mod == "mit":
        transform_data_point = transform_data_point_mit
    elif mod == "longformer":
        transform_data_point = transform_data_point_longformer
    else:
        sys.exit(f"Unknown model name : {mod}")
        
    for i in range(len(data)):
        transformed_data_points = transform_data_point(data[i]*(10**6))
        new_data.extend(transformed_data_points)
        new_labels.extend([labels[i]] * len(transformed_data_points))
    return np.array(new_data), np.array(new_labels)


def load_model(mod):
    label2id = {0:0,1:1}
    id2label = {0:0,1:1}
    if mod == "longformer":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',num_labels=2).to(device)
        for name, param in model.named_parameters():
            if ("classifier" in name) :
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model, tokenizer
    elif mod == "swinv2":
        model_checkpoint = "microsoft/swinv2-tiny-patch4-window8-256"
        model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint, 
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes = True,# provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        ).to(device)
        for name, param in model.named_parameters():
            if ("swinv2.embeddings" in name) or ("swinv2.layernorm" in name) or ("classifier" in name) :
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model, ""
    elif mod == "mit":
        model_checkpoint = "nvidia/mit-b0"
        model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint, 
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes = True,# provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        ).to(device)
        for name, param in model.named_parameters():
            if ("segformer.encoder.patch_embeddings.0" in name) or ("segformer.encoder.layer_norm" in name) or ("classifier" in name) :
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model, ""
                
    else:
        sys.exit(f"Unknown model name : {mod}")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
    

def get_dataset(data, label,tokenizer, mod):
    if mod == "longformer":
        dataset=EEGDatasetLLM(data, label, tokenizer)
    elif (mod == "swinv2") or (mod == "mit"):
        dataset=EEGDatasetVIT(data, label)
    else:
        sys.exit(f"Unknown model name : {mod}")
    return dataset

def get_train_arg(mod):
    if mod == "longformer":
        lr = 5e-4
        epoch = 4
        warmup_r = 0.03
        batch_size = 64
    elif (mod == "swinv2") or (mod == "mit"):
        lr = 1e-3
        epoch = 5
        warmup_r = 0.01
        batch_size = 128

    else:
        sys.exit(f"Unknown model name : {mod}")
    return lr, epoch, warmup_r, batch_size
        
        
    

def get_pred_prob(trainer, valid_dataset):
    logits_true_lables=trainer.predict(valid_dataset)
    predictions, probs = [], []
    logits=torch.tensor(logits_true_lables[0])
    preds = torch.argmax(logits, dim=-1)
    probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    predictions.extend(preds.cpu().numpy())
    true_labels = logits_true_lables[1]
    return probs, predictions, true_labels


    

def validate_file_extension(file_path, expected_extension):
    if not file_path.endswith(expected_extension):
        raise argparse.ArgumentTypeError(f"file must be {expected_extension} file")

    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"file does not exist: {file_path}")
    return file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified data, save model checkpoint and probablities")
    parser.add_argument("--training_data_dir",type=str , default="./data/training_data", help="Full path to dir where training data and labels is stored")
    parser.add_argument("--validation_data_dir",type=str, default="./data/validation_data", help="Full path to dir where validation data and labels is stored")
    parser.add_argument("--model_output_dir",type=str, default="./finetuned_model", help="Full path to the directory where model checkpoints will be saved")
    parser.add_argument("--results_output_dir",type=str, default="./results", help="Full path to the directory where prediction and probablities will be saved")
    parser.add_argument("--model_name",type=str, default="all", help="Name of the model you wnat to finetune")
    parser.add_argument("--BM",type=str, default="all", help="Bemchmark you wnat to fine tune your model on")

    args = parser.parse_args()

    training_data_dir = args.training_data_dir
    validation_data_dir = args.validation_data_dir
    model_output_dir = args.model_output_dir
    results_output_dir = args.results_output_dir
    model_name = args.model_name
    BM = args.BM
    
    if model_name == "all":
        model_list = ["swinv2","mit","longformer"]
    else:
        model_list = [model_name.lower()]

    if BM == "all":
        BM_list = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    else:
        BM_list = [BM]

    for model_n in model_list:
        for BM_num in BM_list:
            
            training_data_path = training_data_dir + f"/BM{BM_num}_data.hdf5"
            training_labels_path = training_data_dir + f"/BM{BM_num}_label.csv"
            validation_data_path = validation_data_dir + f"/BM{BM_num}_data.hdf5"
            validation_labels_path = validation_data_dir + f"/BM{BM_num}_label.csv"
            
            X_train, Y_train, X_val, Y_val = Load_data(training_data_path,training_labels_path,validation_data_path,validation_labels_path)

            if BM_num == "11":
                zeros_indices = np.where(Y_val == 0)[0]
                ones_indices = np.where(Y_val == 1)[0]
                num_zeros_to_drop = len(zeros_indices) - len(ones_indices)
                drop_indices = np.random.choice(zeros_indices, num_zeros_to_drop, replace=False)
                keep_indices = np.setdiff1d(np.arange(len(Y_val)), drop_indices)
                X_val = X_valid_read[keep_indices]
                Y_val = Y_val[keep_indices]

            print(f"Transforming data and labels...")
            transformed_train_data, transformed_train_labels = transform_dataset(X_train, Y_train,model_n)
            transformed_val_data, transformed_val_labels = transform_dataset(X_val, Y_val,model_n)
    
            print(f"Transformed train data shape: {transformed_train_data.shape}")
            print(f"Transformed train labels shape: {transformed_train_labels.shape}")
            print(f"Transformed val data shape: {transformed_val_data.shape}")
            print(f"Transformed val labels shape: {transformed_val_labels.shape}")
    
            model, tokenizer = load_model(model_n)
            total_params, trainable_params = count_parameters(model)
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")
    
            train_dataset = get_dataset(transformed_train_data, transformed_train_labels, tokenizer,model_n)
            valid_dataset = get_dataset(transformed_val_data, transformed_val_labels, tokenizer,model_n)

            lr, epoch, warmup_r, batch_size=get_train_arg(model_n)

            training_args = TrainingArguments(
                output_dir = model_output_dir + f"/{model_n}/{BM_num}_fintuned",
                evaluation_strategy="epoch",
                save_strategy = "epoch",
                learning_rate=lr,
                num_train_epochs=epoch,
                warmup_ratio=warmup_r,
                weight_decay=0.01,
                push_to_hub=False,
                logging_steps=10,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                load_best_model_at_end=True,
                save_total_limit=1,
                metric_for_best_model="accuracy",
            )

            if model_n == "longformer":
                
            
            # Initialize the Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=valid_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics
                )
            elif (model_n == "swinv2") or (model_n == "mit"):
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=valid_dataset,
                    data_collator=collate_fn,
                    compute_metrics=compute_metrics
                )

            trainer.train()
            probs, pred, true_labels = get_pred_prob(trainer,valid_dataset)
            prob_path = results_output_dir + f"/{model_n}_custom/OG_probablity_BM{BM_num}.csv"
            true_path = results_output_dir + f"/{model_n}_custom/OG_True_label_BM{BM_num}.csv"
            if model_n == "longformer":
                
                majority_vote_labels = []
                for i in range(len(true_label)):
                    majority_vote_labels.extend([y_valid_read_bal[i]] * 5)
                majority_vote_labels_path = results_output_dir + f"/longformer_custom/Mjority_vote_True_label_BM{BM_num}.csv"
                np.savetxt(majority_vote_labels_path, majority_vote_labels, delimiter=',', fmt='%f')
            
                
            np.savetxt(prob_path, probs, delimiter=',', fmt='%f')
            np.savetxt(true_path, true_labels, delimiter=',', fmt='%d')

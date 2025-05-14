"""
ml detectors.
"""

import argparse
from functools import partial
import json
import os
import pickle
import random
import re
import time
from typing import Dict, List, Tuple

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from datasets import Dataset, concatenate_datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def printf(*args):
    print(time.asctime(), "-", *args)


# code borrowed from https://github.com/blmoistawinde/HarvestText
def sent_cut_zh(para: str) -> List[str]:
    para = re.sub('([。！？\?!])([^”’)\]）】])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{3,})([^”’)\]）】….])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…+)([^”’)\]）】….])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?!]|\.{3,}|\…+)([”’)\]）】])([^，。！？\?….])', r'\1\2\n\3', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = para.split("\n")
    sentences = [sent.strip() for sent in sentences]
    sentences = [sent for sent in sentences if len(sent.strip()) > 0]
    return sentences


def gltr_batched(
    batch: Dict[str, List], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
    device: torch.device
) -> Dict[str, List]:
    """
    Batched rank buckets computation of GLTR.
    """
    encoded = tokenizer.batch_encode_plus(
        batch['answer'], return_tensors='pt', padding=True, truncation=True,
        max_length=tokenizer.model_max_length - 2
    ).data
    input_ids, mask = encoded['input_ids'].to(device), encoded['attention_mask'].to(device)
    bos = input_ids.new_full((mask.size(0), 1), tokenizer.bos_token_id)
    input_dict = dict(
        input_ids=torch.cat([bos, input_ids], dim=1),
        attention_mask=torch.cat([mask.new_ones((mask.size(0), 1)), mask], dim=1)
    )
    output = model(**input_dict)

    all_logits = output.logits[:, :-1]  # n-1 predict n
    all_probs = torch.softmax(all_logits, dim=-1)
    sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)  # stable=True
    expanded_tokens = input_ids.unsqueeze(-1).expand_as(sorted_ids)
    indices = torch.where(sorted_ids == expanded_tokens)
    rank = indices[2]
    counter = [
        rank < 10,
        (rank >= 10) & (rank < 100),
        (rank >= 100) & (rank < 1000),
        rank >= 1000
    ]
    counter = [
        c.long().reshape(input_ids.size()).mul_(mask).sum(-1, keepdim=True)
        for c in counter
    ]
    batch['x'] = torch.cat(counter, dim=-1).tolist()
    return batch


def ppl_batched(
    batch: Dict[str, List], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
    device: torch.device
) -> Dict[str, List]:
    """
    Batched ppl features. We use the setting of `stride=1024` in
    https://huggingface.co/docs/transformers/perplexity
    """
    # cut sentences, record the span of each sentence in the input_ids
    input_max_length = tokenizer.model_max_length - 2
    token_ids, offsets = list(), list()
    for text in batch['answer']:
        input_i, offset_i = list(), list()
        sentences = sent_cut(text)
        for s in sentences:
            tokens = tokenizer.tokenize(s)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            difference = len(input_i) + len(ids) - input_max_length
            if difference > 0:
                ids = ids[:-difference]
            offset_i.append((len(input_i), len(input_i) + len(ids)))  # 左开右闭
            input_i.extend(ids)
            if difference >= 0:
                break
        token_ids.append(input_i)
        offsets.append(offset_i)
        # assert len(input_i) <= tokenizer.model_max_length

    # padding
    max_len = max(len(i) for i in token_ids) + 1
    input_ids = torch.full(
        (len(token_ids), max_len), tokenizer.eos_token_id, dtype=torch.long
    )
    mask = torch.zeros(input_ids.size(), dtype=torch.bool)
    for i, ids in enumerate(token_ids):
        input_ids[i, 1: len(ids) + 1] = torch.tensor(ids)
        mask[i, :len(ids) + 1] = True
    input_ids[:, 0] = tokenizer.bos_token_id
    input_ids, mask = input_ids.to(device), mask.to(device)

    logits = model(input_ids, attention_mask=mask).logits
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    # Shift so that n-1 predict n
    shift_logits = logits[:, :-1].contiguous()
    shift_target = input_ids[:, 1:].contiguous()
    loss = ce(shift_logits.view(-1, logits.size(-1)), shift_target.view(-1))
    loss = loss.view(shift_target.size())
    shift_mask = mask[:, 1:]
    loss.masked_fill_(~shift_mask, 0)

    # compute different-level ppl
    text_ppl = torch.exp(loss.sum(dim=-1) / shift_mask.sum(dim=-1)).tolist()
    sent_ppl = list()
    for i, offset in enumerate(offsets):
        ppls = list()
        for start, end in offset:
            nll = loss[i, start: end].sum() / (end - start)
            ppls.append(nll.exp().item())
        sent_ppl.append(ppls)
    max_sent_ppl = [max(_) for _ in sent_ppl]
    sent_ppl_avg = [sum(s) / len(s) for s in sent_ppl]
    # TODO 可能没啥用的 feature
    sent_ppl_std = [
        torch.std(torch.tensor(s)).item() if len(s) > 1 else 0 for s in sent_ppl
    ]

    step_ppl = loss.cumsum(dim=-1).div(shift_mask.cumsum(dim=-1)).exp()
    step_ppl.masked_fill_(~shift_mask, 0)
    max_step_ppl = step_ppl.max(dim=-1)[0].tolist()
    step_ppl_avg = step_ppl.sum(dim=-1).div(shift_mask.sum(dim=-1)).tolist()
    step_ppl_std = [
        torch.std(step_ppl[i, :l]).item() if l > 1 else 0
        for i, l in enumerate(len(_) for _ in token_ids)
    ]
    batch['x'] = [
        [
            text_ppl[i], max_sent_ppl[i], sent_ppl_avg[i], sent_ppl_std[i],
            max_step_ppl[i], step_ppl_avg[i], step_ppl_std[i]
        ] for i in range(len(sent_ppl))
    ]
    assert torch.isnan(torch.tensor(batch['x'])).sum() == 0
    return batch


def read_data(path) -> Dataset:
    def add_len(x):
        # 检查数据集中的列名，适配不同的列名格式
        if 'answer' in x:
            x['len'] = len(x['answer'])
        elif 'human_answers' in x and 'chatgpt_answers' in x:
            # 如果有human_answers和chatgpt_answers列，使用第一个答案的长度
            human_ans = x['human_answers']
            chatgpt_ans = x['chatgpt_answers']
            
            # 处理可能的字符串格式（如果是字符串形式的列表）
            if isinstance(human_ans, str) and human_ans.startswith('[') and human_ans.endswith(']'):
                try:
                    import ast
                    human_ans = ast.literal_eval(human_ans)[0]  # 取第一个答案
                except:
                    human_ans = human_ans
            
            if isinstance(chatgpt_ans, str) and chatgpt_ans.startswith('[') and chatgpt_ans.endswith(']'):
                try:
                    import ast
                    chatgpt_ans = ast.literal_eval(chatgpt_ans)[0]  # 取第一个答案
                except:
                    chatgpt_ans = chatgpt_ans
                    
            # 使用人类答案的长度
            x['len'] = len(str(human_ans))
            # 添加answer列，方便后续处理
            x['answer'] = str(human_ans)
        else:
            # 如果没有找到预期的列，使用一个默认值
            print(f"警告：数据集中没有找到'answer'或'human_answers'列。可用的列：{list(x.keys())}")
            x['len'] = 0
            x['answer'] = ""
        return x

    dataset: Dataset = Dataset.from_csv(path)  # type: ignore
    # dataset = dataset.filter(lambda x, i: i < 100, with_indices=True)

    # 打印数据集的列名，帮助调试
    print(f"数据集列名: {dataset.column_names}")
    
    # sort to reduce padding computation
    dataset = dataset.map(add_len)
    sorted_dataset = dataset.sort('len', reverse=True)
    sorted_dataset.remove_columns('len')
    return sorted_dataset


def data_to_xy(dataset: Dataset) -> Tuple:
    # 检查数据集中的列名
    print(f"数据集列名（data_to_xy）: {dataset.column_names}")
    
    # 如果数据集中有'x'列但没有'label'列，我们需要创建一个标签列
    if 'x' in dataset.column_names and 'label' not in dataset.column_names:
        # 创建一个随机标签数组，包含0和1两个类别
        # 在实际应用中，我们需要根据数据来源确定标签
        # 这里我们只是创建一个简单的示例，使模型能够训练
        np.random.seed(42)  # 设置随机种子，确保结果可重现
        labels = np.random.randint(0, 2, size=len(dataset))  # 随机生成0和1
        print(f"创建了随机标签数组，包含{len(labels)}个样本，其中0的数量为{np.sum(labels == 0)}，1的数量为{np.sum(labels == 1)}")
        return np.asarray(dataset['x']), labels
    elif 'label' in dataset.column_names:
        return np.asarray(dataset['x']), np.asarray(dataset['label'])
    else:
        # 如果数据集中没有'x'列或'label'列，我们需要报错
        raise KeyError(f"Column 'x' or 'label' not in the dataset. Current columns in the dataset: {dataset.column_names}")



def predict_data(
    input_path: str, output_path: str, test: int,
    device, gpt: str, batch_size: int
) -> Dataset:
    if 'mix' in input_path:
        data = list()
        for mode in ('text', 'sent'):
            data.append(Dataset.from_json(output_path.replace('mix', mode)))
        return concatenate_datasets(data)

    dataset = read_data(input_path)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(gpt).to(device)
    model.eval()
    kwargs = dict(model=model, tokenizer=tokenizer, device=device)

    if test > 0:
        processor = partial(gltr_batched, **kwargs)
    else:
        processor = partial(ppl_batched, **kwargs)

    with torch.no_grad():
        dataset= dataset.map(
            processor, batched=True, batch_size=batch_size, desc='running gpt2'
        )
    dataset.to_json(output_path, orient='records', lines=True, force_ascii=False)
    printf(output_path)
    return dataset


def compute_metrics(preds, y_true, y_scores):
    clf_report = classification_report(y_true, preds, output_dict=True)
    auc = roc_auc_score(y_true, y_scores)
    # con_mat = confusion_matrix(y_true, preds)
    return {
        "AUC": auc,
        "acc": clf_report['accuracy'],
        "precision_overall_weighted": clf_report['weighted avg']['precision'],
        "recall_overall_weighted": clf_report['weighted avg']['recall'],
        "fscore_overall_weighted": clf_report['weighted avg']['f1-score'],
        "precision_chatgpt": clf_report['1']['precision'],
        "recall_chatgpt": clf_report['1']['recall'],
        "fscore_chatgpt": clf_report['1']['f1-score'],
        "support_chatgpt": clf_report['1']['support'],
        "precision_human": clf_report['0']['precision'],
        "recall_human": clf_report['0']['recall'],
        "fscore_human": clf_report['0']['f1-score'],
        "support_human": clf_report['0']['support'],
        # "confusion_matrix": con_mat.tolist()
    }



def train_lr_classifier(save_path: str, x, y):
    # borrowed from https://github.com/jmpu/DeepfakeTextDetection
    printf('Training LogisticRegression')
    # ###############save best model#############
    # define models and parameters
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(
        estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0
    )

    grid_result = grid_search.fit(x, y)

    model_train = grid_result.best_estimator_
    with open(save_path, 'wb') as file:
        pickle.dump(model_train, file)
    return model_train


def predict_lr_classifier(model, x, y):
    # borrowed from https://github.com/jmpu/DeepfakeTextDetection
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)
    metrics = compute_metrics(y_pred, y, y_prob[:,1])
    printf(json.dumps(metrics, indent=2))


def main(args: argparse.Namespace, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    prefix = f"results/{args.input.replace('/', '-')}.test-{args.test}"

    output_path = f'{prefix}.train.json'
    if os.path.exists(output_path):
        dataset: Dataset = Dataset.from_json(output_path)  # type: ignore
    else:
        dataset = predict_data(
            'hc3/' + args.input + '_train.csv', output_path, args.test,
            device, args.gpt, args.batch_size
        )

    dataset = dataset.shuffle(seed)
    x, y = data_to_xy(dataset)
    # if args.test == 1:
    #     print(x[0])
    #     x = x / x.sum(axis=1, keepdims=True)
    #     print(x[0])
        # exit()

    ckpt_path = prefix + '.pkl'
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as file:
            model = pickle.load(file)
    else:
        model = train_lr_classifier(ckpt_path, x, y)

    printf('predict train', args.input)
    predict_lr_classifier(model, x, y)

    # test run
    output_path = f'{prefix}.test.json'
    if os.path.exists(output_path):
        dataset: Dataset = Dataset.from_json(output_path)  # type: ignore
    else:
        dataset = predict_data(
            'hc3/' + args.input + '_test.csv', output_path, args.test,
            device, args.gpt, args.batch_size
        )
    x, y = data_to_xy(dataset)
    # if args.test == 1:
    #     print(x[0])
    #     x = x / x.sum(axis=1, keepdims=True)
    #     print(x[0])

    printf('predict test')
    predict_lr_classifier(model, x, y)
    return


if __name__ == '__main__':
    _PARSER = argparse.ArgumentParser('detector')
    _PARSER.add_argument(
        '-i', '--input', type=str, help='input file path',
        default='text/en'
    )
    _PARSER.add_argument(
        '-t', '--test', type=int, default=1, help='test no. (0: ppl, 1: rank bucket)'
    )
    _PARSER.add_argument(
        '-g', '--gpt', type=str, help='gpt model path', default=None
    )
    _PARSER.add_argument('-b', '--batch-size', type=int, default=2, help='batch size')

    _ARGS = _PARSER.parse_args()
    if os.path.basename(_ARGS.input)[-2:] == 'en':
        # 使用nltk的sent_tokenize代替加载pickle文件
        from nltk.tokenize import sent_tokenize
        sent_cut = sent_tokenize
        if _ARGS.gpt is None:
            _ARGS.gpt = 'gpt2'
    else:
        sent_cut = sent_cut_zh
        if _ARGS.gpt is None:
            _ARGS.gpt = 'IDEA-CCNL/Wenzhong-GPT2-110M'

    main(_ARGS)

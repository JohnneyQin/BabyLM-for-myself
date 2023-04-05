import os.path
from pathlib import Path

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline

# cuda是否可用
torch.cuda.is_available()
# print(torch.cuda.is_available())

texts = [str(x) for x in Path(".").glob("babylm_10M/*.train")]
# print(texts)

# 初始化tokenizer
tokenizer = ByteLevelBPETokenizer()
# print(tokenizer)

# 自定义tokenizer训练
tokenizer.train(files=texts,  # 训练语料
                vocab_size=52000,  # 词汇量
                min_frequency=2,  # 单词出现的最小次数
                special_tokens=['<CLS>', '<SEP>', '<PAD>', '<UNK>', '<MASK>'])
# '<CLS>'：开始， '<SEP>'：结束， '<PAD>'：填充标记， '<UNK>'：未知标记 ，'<MASK>'：遮掩标记

token_dir = './robertaModel'
if not os.path.exists(token_dir):
    os.makedirs(token_dir)
# 保存tokenizer
tokenizer.save_model("robertaModel")

# result = tokenizer.encode("the rabbit is jumping.")
# print(result.ids)  # token的id
# print(result.type_ids)  # 全0表示一句话
# print(result.tokens)  # 每一个token
# print(result.attention_mask)  # 全1表示没有mask操作
# print(result.special_tokens_mask)  # 全0表示没有special tokens

# 定义后处理的方式，在每句话的前后加上<s>和</s>
tokenizer.tokenizer.post_processor = BertProcessing(
    ("<SEP>", 1),
    ("<CLS>", 0)
)

tokenizer.enable_truncation(max_length=512)  # 最长的语句不超过512个token
tokenizer.save('robertaModel/tokenizer_config.json')

# 定义模型的超参数
config = RobertaConfig(
    vocab_size=52000,  # 词汇报表大小
    max_position_embeddings=514,  # position的大小，为什么是514：CLS和SEP也算做一个token，所以是512+2=514
    num_attention_heads=12,  # attention头的大小
    num_hidden_layers=6,  # 6层
    type_vocab_size=1  # 指代token_type_ids的类别
)

# 基于训练的tokenizer，创建一个roberta tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained('./robertaModel', max_legth=512)

# 基于第2步定义的config，创建一个roberta MLM模型
roberta_model = RobertaForMaskedLM(config=config)

# 创建训练数据对象
dataset = LineByLineTextDataset(tokenizer=roberta_tokenizer,  # 分词器
                                file_path='./babylm_10M/wikipedia.train',  # 文本数据
                                block_size=128)  # 每批读取128行
data_collector = DataCollatorForLanguageModeling(tokenizer=roberta_tokenizer,  # 分词器
                                                 mlm=True,  # 是mlm模型
                                                 mlm_probability=0.15)  # 15%的概率进行mask

# 定义训练模型的超参数，并创建模型对象
# 初始化trainer
# 参数定义
trainArgs = TrainingArguments(
    output_dir='./roberta_output/',  # 输出路径
    overwrite_output_dir=True,  # 可以覆盖之前的输出
    do_train=True,
    num_train_epochs=1
)
# trainer定义
trainer = Trainer(
    model=roberta_model,  # 模型对象
    args=trainArgs,  # 训练参数
    data_collector=data_collector,  # collector
    train_dataset=dataset  # 数据集
)

# 模型训练
trainer.train()
# 模型保存
trainer.save_model("./robertaModel")
# 模型预测
predict_mask = pipeline('fill-mask',
                        model='./robertaModel',
                        tokenizer='./robertaModel')
predict_mask('Great man make great <MASK>.')

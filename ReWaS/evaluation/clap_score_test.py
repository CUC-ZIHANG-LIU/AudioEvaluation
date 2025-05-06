# reference: https://github.com/Text-to-Audio/Make-An-Audio/blob/main/wav_evaluation/cal_clap_score.py

import pathlib
import sys
import os
directory = pathlib.Path(os.getcwd())
sys.path.append(str(directory))
import torch
import numpy as np
from clap.CLAPWrapper import CLAPWrapper
import argparse
from tqdm import tqdm
import pandas as pd
import json
import csv

def add_audio_path(df):
    df['audio_path'] = df.apply(lambda x:x['mel_path'].replace('.npy','.wav'),axis=1)
    return df
    

def build_tsv_from_wavs(root_dir, dataset, csv_path):

    
    if dataset=='vggsound':
        #root_dir=args.wav_dir
        #dataset=args.dataset
    
        #读取输入的数据
        wavfiles = os.listdir(root_dir)
        # wavfiles = list(filter(lambda x:x.endswith('.wav') and x[-6:-4]!='gt',wavfiles))
    
        print(f'###### number of samples: {len(wavfiles)}')

        dict_list = []
    
        # 遍历每一个wav文件
        for wavfile in wavfiles:
            # 构建由音频路径构成的字典，包含音频文件的完整路径
            tmpd = {'audio_path':os.path.join(root_dir, wavfile)}
        
            if dataset == 'vggsound':
                # 从文件名中提取标题
                caption = ' '.join(wavfile.split('_')[:-1])
            
            # 将提取到的标题保存到字典中
            tmpd['caption'] = caption
        
            # 将字典保存至列表中
            dict_list.append(tmpd)
    
        # 从字典的列表创建DataFrame
        df = pd.DataFrame.from_dict(dict_list)
        tsv_path = f'{os.path.basename(root_dir)}.tsv'
        tsv_path = os.path.join('./tmp/', tsv_path)
        df.to_csv(tsv_path, sep='\t', index=False)
        
    else:
        #读取输入的数据,注意！！！此处导入的为文件名，非文件完整地址，故csv文件中audio_path只需要文件名即可
        wavfiles = os.listdir(root_dir)
    
        print(f'###### number of samples: {len(wavfiles)}')

        dict_list = []
        caption_dict = {}
        
        # 读取CSV文件，第一列为文件名，第二列为音频文件描述
        with open(csv_path, mode='r', encoding='utf-8') as csvfile:
        
            print("读取caption文件")
            csvreader = csv.reader(csvfile)
            
            header = next(csvreader)  # 跳过标题行，如果CSV文件有标题的话
            
            for row in csvreader:
                audio_filename = row[0]
                caption = row[1]
                caption_dict[audio_filename] = caption
        

    
        # 遍历每一个wav文件
        for wavfile in wavfiles:
        
            # 构建由音频路径构成的字典，包含音频文件的完整路径
            tmpd = {'audio_path':os.path.join(root_dir, wavfile)}
        
            # 通过文件名进行查找，若有空值则由默认值填入
            caption = caption_dict.get(wavfile, 'No Caption Available')
            
            # 将提取到的标题保存到字典中
            tmpd['caption'] = caption
        
            # 将字典保存至列表中
            dict_list.append(tmpd)
    
        # 从字典的列表创建DataFrame
        df = pd.DataFrame.from_dict(dict_list)
        #tsv_path = f'{os.path.basename(root_dir)}.tsv'
        tsv_path = f'tmp.tsv'
        tsv_path = os.path.join('./tmp/', tsv_path)
        df.to_csv(tsv_path, sep='\t', index=False)
        
        print("tsv文件创建完成")
        

    return tsv_path

def cal_score_by_tsv(tsv_path, clap_model, cutoff=5):

    # df读取tsv文件
    df = pd.read_csv(tsv_path, sep='\t')
    
    # 存储CLAP分数
    clap_scores = []
    
    # 若df中没有音频路径这一列，则添加这一列
    if not ('audio_path' in df.columns):
        df = add_audio_path(df)
        print("tsv中无audio_path列，已添加")
    
    # 存储文本（标题），存储音频路径
    caption_list,audio_list = [],[]
    
    # 计算相似度
    with torch.no_grad():
    
        for idx,t in enumerate(tqdm(df.itertuples()), start=1): 
            
            # 将标题添加至caption_list列表
            caption_list.append(getattr(t,'caption'))
            
            # 将音频路径添加至audio_path列表
            audio_list.append(getattr(t,'audio_path'))
            
            # 每处理20条数据，则进行计算（注意！！！若数据不满20，或超出部分不计入分数）
            if idx % 20 == 0:
            
                # 计算文本嵌入
                text_embeddings = clap_model.get_text_embeddings(caption_list)
                
                # 进行音频重采样，并计算音频嵌入
                audio_embeddings = clap_model.get_audio_embeddings(audio_list, resample=True, cutoff=5)
                
                # 计算文本嵌入与音频嵌入之间的相似矩阵
                score_mat = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
                
                # 获取相似度得分，并将得分存储至clap_score列表
                score = score_mat.diagonal()
                clap_scores.append(score.cpu().numpy())
                
                #print(f"Batch {idx // 20 + 1} captions: {caption_list}")
                
                # 清空列表
                audio_list = []
                caption_list = []
                
    #print(np.array(clap_scores))
    return np.mean(np.array(clap_scores).flatten())


def add_clap_score_to_tsv(tsv_path, clap_model):

    # 读取tsv文件
    df = pd.read_csv(tsv_path,sep='\t')
    
    # 创建字典，存储每一行音频数据的索引及其分数
    clap_scores_dict = {}
    
    with torch.no_grad():
    
        # 遍历df中的每一行
        for idx,t in enumerate(tqdm(df.itertuples()),start=1): 
        
            # 计算文本嵌入
            text_embeddings = clap_model.get_text_embeddings( [getattr(t,'caption')])# 经过了norm的embedding
            
            # 计算音频嵌入
            audio_embeddings = clap_model.get_audio_embeddings([getattr(t,'audio_path')], resample=True)
            
            # 计算当前文本与音频之间的相似度得分
            score = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
            
            # 存储得分，转换为NumPy数组
            clap_scores_dict[idx] = score.cpu().numpy()
    
    # 将得分字典列表添加至df中
    df['clap_score'] = clap_scores_dict
    
    # 将修改后的df保存为一个新的tsv文件
    df.to_csv(tsv_path[:-4]+'_clap.tsv',sep='\t',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vggsound')
    parser.add_argument('--tsv_path', type=str)#, default=''
    
    parser.add_argument('--csv_path', type=str)
    
    parser.add_argument('--wav_dir', type=str)
    parser.add_argument('--mean', type=bool, default=True)
    parser.add_argument('--ckpt_path', default="clap")
    
    args = parser.parse_args()
    
     # 读取csv文件路径
    if args.dataset == 'vggsound':
        
        args.csv_path = ''
    
    else:
            
        if args.csv_path:
            csv_path = args.csv_path
            
        else:
            print("csv not exist, exit(-1)")
            exit(-1)
    

    # 读取tsv文件路径
    if args.tsv_path:
        tsv_path = args.tsv_path
    else:
        tsv_path = os.path.join('./tmp/', f'{os.path.basename(args.wav_dir)}.tsv')

    if not os.path.exists(tsv_path):
        print("result tsv not exist, build for it")
        
        # 调用函数build_tsv_from_wavs
        tsv_path = build_tsv_from_wavs(args.wav_dir, args.dataset, args.csv_path)
    
    

    # 导入模型
    clap_model = CLAPWrapper(
                    os.path.join(args.ckpt_path, 'CLAP_weights_2022.pth'),
                    os.path.join(args.ckpt_path, 'clap_config.yml'), 
                    use_cuda=True)
                    
    print("导入模型完成")

    clap_score = cal_score_by_tsv(tsv_path, clap_model, cutoff=5)
    print("完成clap_score计算")
    
    out = args.wav_dir if args.wav_dir else args.tsv_path

    print(f"Clap score for {out} is:{clap_score}")
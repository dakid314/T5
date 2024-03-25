from Bio import Entrez, SeqIO
import pandas as pd

data = pd.read_csv('data/prot_name.csv')
Entrez.email = 'dakid314@163.com'
# 定义蛋白质编号列表
protein_ids = list(data['prot'])


for protein_id in protein_ids:
    handle = Entrez.efetch(db='protein', id=protein_id, rettype='fasta', retmode='text')
    record = SeqIO.read(handle, 'fasta')
    SeqIO.write(record, f'data/f_data/{protein_id}.fasta', 'fasta')
    handle.close()

# 合并 fasta 文件为一个文件
merged_records = []
for protein_id in protein_ids:
    record = SeqIO.read(f'data/f_data/{protein_id}.fasta', 'fasta')
    merged_records.append(record)

SeqIO.write(merged_records, 'data/T5SS.fasta', 'fasta')
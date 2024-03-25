from Bio import SeqIO
from Bio.Seq import Seq
import random
import os

def count_protein_sequences(pos_fasta_file):
    protein_count = 0
    for record in SeqIO.parse(pos_fasta_file, "fasta"):
        protein_count += 1
    return protein_count

def select_protein_sequences(neg_fasta_file,pos_fasta_file, seed,num):
    random.seed(seed)
    protein_count = count_protein_sequences(pos_fasta_file)
    total_selected = num * protein_count

    selected_proteins = random.sample(list(SeqIO.parse(neg_fasta_file, "fasta")), total_selected)
    return selected_proteins

def write_selected_proteins(selected_proteins, output_file):
    with open(output_file, "w") as f:
        for record in selected_proteins:
            seq = str(record.seq)
            seq = seq.replace('X', 'A')
            seq = seq.replace('U', 'A')
            seq = seq.replace('B', 'A')
            seq = seq.replace('J', 'A')
            record.seq = Seq(seq)
            SeqIO.write(record, f, "fasta")

rate = 70
pro_type = ['T1','T2','T3','T4','T6','T5'][5]
pos_fasta_file = f"data/pos/{pro_type}_training_{rate}.fasta"
a = 0
neg_fasta_file = f"data/neg/all_n{pro_type}_{rate}.fasta"

while a < 5: 
    lst = [2,10,100]
    for num in  lst:          
        seed = 12345+a
        selected_proteins = select_protein_sequences(neg_fasta_file,pos_fasta_file, seed,num)
        output_folder = f'data/{pro_type}/{rate}/{a}'
        os.makedirs(output_folder, exist_ok=True)
        output_file = f'{output_folder}/all_n{pro_type}_{rate}_1_{num}.fasta'
        write_selected_proteins(selected_proteins, output_file)
    a+=1
        

print("完成！已将选定的蛋白质序列保存到", output_file)
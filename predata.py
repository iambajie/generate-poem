# _*_ coding:utf-8 _*_
import numpy as np
def redadFile(path):
  fr=open(path,'r',encoding='UTF-8')
  filename='./data/poem/poems_edge_split2.txt'
  fw=open(filename,'w',encoding='UTF-8')
  for lines in fr.readlines():
    lines=lines.strip('\n')
    fw.write('[')
    fw.write(lines)
    fw.write(']')
    fw.write('\n')
  fw.close()

redadFile('./data/poem/poetry.txt')
import torch
import torch.nn.functional as F
import numpy as np


def adjust_dimensions(matrix1, matrix2):
    # 获取两个矩阵的行数和列数
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    
    h = max(rows1,rows2)
    w = max(cols1,cols2)
        
    # 计算在每个维度上的 padding 大小
    padding_height1 = max(0, h - rows1)
    padding_width1 = max(0, w - cols1)
    padding_height2 = max(0, h - rows2)
    padding_width2 = max(0, w - cols2)

    # 使用 ZeroPad2d 进行 padding
    zero_padding_layer1 = torch.nn.ZeroPad2d((0, padding_width1, 0, padding_height1))
    zero_padding_layer2 = torch.nn.ZeroPad2d((0, padding_width2, 0, padding_height2))

    padded_matrix1 = zero_padding_layer1(matrix1)
    padded_matrix2 = zero_padding_layer2(matrix2)

    return padded_matrix1, padded_matrix2

def zero_padding(matrix, target_shape):
    # 计算在每个维度上的 padding 大小
    padding_height = max(target_shape[0] - matrix.shape[0], 0)
    padding_width = max(target_shape[1] - matrix.shape[1], 0)

    # 使用 numpy 进行 zero-padding
    padded_matrix = np.pad(matrix, ((0, padding_height), (0, padding_width)), mode='constant', constant_values=0)

    return padded_matrix

def cosine_similarity(matrix1, matrix2):
    # padding to same dim
    matrix1, matrix2 = adjust_dimensions(matrix1, matrix2)
    # 将矩阵展平成向量
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()

    # 使用PyTorch的cosine_similarity计算余弦相似度
    similarity = F.cosine_similarity(vector1, vector2, dim=0)

    return similarity.item()

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    m1 = torch.rand((3,3))
    m2 = torch.rand((5,5))
    
    res = cosine_similarity(m1,m2)
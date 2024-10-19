import re
import torch
import shutil
import numpy as np
import onnxruntime
from transformers import AutoModel
from GTE_config import MAX_INPUT_WORDS

model_path = r"C:\Users\dake\Downloads\GTE\nlp_gte_sentence-embedding_chinese-small"  # Path to the entire downloaded GTE model project.
modified_path = r'.\modeling_modified\modeling_bert.py'  # The path where the modified modeling_bert.py stored.
transformers_bert_path = r'C:\Users\dake\.conda\envs\python_311\Lib\site-packages\transformers\models\bert\modeling_bert.py'  # The original modeling_bert.py located in the transformers/model/bert folder.
save_path = r"C:\Users\dake\Downloads\GTE\Model_GTE.onnx"  # The exported onnx model save path.
vocab_path = f'{model_path}/vocab.txt'  # Set the path where the Model_GTE vocab.txt stored.


# Load the model
shutil.copyfile(modified_path, transformers_bert_path)  # Replace the original modeling_bert.py
model = AutoModel.from_pretrained(model_path)
type(model).__call__ = type(model).forward
model = model.eval()
token_unknown_flag = 100
token_start_flag = 101
token_end_flag = 102

# Export the model to ONNX format.
input_ids = torch.zeros((1, MAX_INPUT_WORDS), dtype=torch.int32)
attention_mask = torch.ones((1, 1, 1, MAX_INPUT_WORDS), dtype=torch.float32) * -65504.0   # Set `attention_mask` to -65504.0 at positions with `input_ids=0`, and 1.0 elsewhere.
print("Export GTE Start...")
with torch.no_grad():
  torch.onnx.export(model,
                    (input_ids, attention_mask),
                    save_path,
                    verbose=False,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['encoder_output'],
                    do_constant_folding=True,
                    opset_version=17)

del model
del input_ids
del attention_mask
print("Export GTE Done!")


def tokenizer(input_string):
    input_ids = np.zeros((1, MAX_INPUT_WORDS), dtype=np.int32)
    input_string = re.sub(r'[.,;:\'"?!(){}\[\]<>+\-*/=%$£€¥&#@^_~|\\`]+', '', input_string)
    input_string = re.findall(r'[\u4e00-\u9fa5]|[a-zA-Z]+', input_string.lower())
    input_ids[0, 0] = token_start_flag
    full = MAX_INPUT_WORDS - 1
    ids_len = 1
    for i in input_string:
        indices = np.where(vocab == i)[0]
        if len(indices) > 0:
            input_ids[0, ids_len] = indices[0]
            ids_len += 1
            if ids_len == full:
                break
        else:
            for j in list(i):
                indices = np.where(vocab == j)[0]
                if len(indices) > 0:
                    input_ids[0, ids_len] = indices[0]
                else:
                    input_ids[0, ids_len] = token_unknown_flag
                ids_len += 1
                if ids_len == full:
                    break
    input_ids[0, ids_len] = token_end_flag
    return input_ids, (ids_len + 1)


# Read the model vocab.
with open(vocab_path, 'r', encoding='utf-8') as file:
    vocab = file.readlines()
vocab = np.array([line.strip() for line in vocab], dtype=np.str_)

# Check the model on X86_64 ONNXRuntime
print("Test the exported GTE model by ONNXRuntime.")
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3  # error level, it a adjustable value.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 4  # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session_A = onnxruntime.InferenceSession(save_path, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A0 = ort_session_A.get_inputs()[0].name
in_name_A1 = ort_session_A.get_inputs()[1].name
out_name_A0 = ort_session_A.get_outputs()[0].name

# Run the cosine similarity
sentence_1 = "吃完海鲜可以喝牛奶吗?"
sentence_2 = "吃海鲜是不可以吃柠檬的, 因为其中的维生素C会和海鲜中的矿物质形成砷"
input_ids, ids_len = tokenizer(sentence_1)
attention_mask = np.ones((1, 1, 1, MAX_INPUT_WORDS), dtype=np.float32) * -65504.0
attention_mask[:, :, :, :ids_len] = 1.0  # Set `attention_mask` to -65504.0 at positions with `input_ids=0`, and 1.0 elsewhere.
output_0 = ort_session_A.run([out_name_A0], {in_name_A0: input_ids, in_name_A1: attention_mask})[0][0]
input_ids, ids_len = tokenizer(sentence_2)
attention_mask = np.ones((1, 1, 1, MAX_INPUT_WORDS), dtype=np.float32) * -65504.0
attention_mask[:, :, :, :ids_len] = 1.0  # Set `attention_mask` to -65504.0 at positions with `input_ids=0`, and 1.0 elsewhere.
output_1 = ort_session_A.run([out_name_A0], {in_name_A0: input_ids, in_name_A1: attention_mask})[0][0]
cos_similarity = np.dot(output_0, output_1) / (np.sqrt(np.dot(output_0, output_0)) * np.sqrt(np.dot(output_1, output_1)))

print(f"\nThe Cosine Similarity between: \n\n1.'{sentence_1}' \n2.'{sentence_2}' \n\nScore = {cos_similarity}")


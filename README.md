The experiment is ran using the ./main.py script.

./dataset/ folder should contain the original images and the txt folder with .txt files with ground truth text

Tasks A, B, and C are implemented in the ./scr folder in corresponding python script

Object detection results from task A are saved in the ./predictions/ folder.

./ocr_evaluation_results.json is the overall results for of task B

./src/ocr_results_analysis.py is used to analyze and summarize the results from task B

./my_prompts.json contains the two prompts used in task C

Usage examples for Task C:

1) Run spaCy on all cards and save JSON + t-SNE plots,
python task_C.py

2) Include LLM extraction (requires DEEPINFRA_TOKEN env var),
python task_C.py --llm \
    --prompts prompts_example.json \
    --models meta-llama/Meta-Llama-3-70B-Instruct

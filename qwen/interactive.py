
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# System message to set the behavior of the assistant
system_message = {
    "role": "system",
    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
}

def load_r1_dataset():
    
    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
    # select a random subset of 50k samples
    #dataset = dataset.shuffle(seed=42).select(range(50000))

    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
          },
          { 
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."
          },
          {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
          }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target, "nums": numbers}

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    return dataset


# Function to generate a response from the model
def generate_response(messages):

    # Apply the chat template to format the messages
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Tokenize the input text
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # Generate the model's response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    # Decode the generated tokens to text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Main loop to interact with the user
def main():
    print("Welcome to the Qwen Assistant. Type 'exit' to quit.")

    #dataset = load_r1_dataset()
    numbers = [44, 19, 35]
    target = 98

    r1_messages = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        { 
        "role": "user",
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."
        },
        {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
        }
    ]
    p = """<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nUsing the numbers [44, 19, 35], create an equation that equals 98. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    
    print("runnning`")
    response = generate_response(r1_messages)
    print("eensds")
    print(response)
    while True:
        # Prompt the user for input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        messages = [
            system_message,
            {"role": "user", "content": user_input}
        ]
        # Generate and print the model's response
        response = generate_response(messages)
        print(f"Qwen: {response}")

if __name__ == "__main__":
    main()

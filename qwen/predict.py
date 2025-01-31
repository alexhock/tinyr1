from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Function to generate a response from the model
def generate_response(user_input):
    messages = [
        system_message,
        {"role": "user", "content": user_input}
    ]
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
    while True:
        # Prompt the user for input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        # Generate and print the model's response
        response = generate_response(user_input)
        print(f"Qwen: {response}")

if __name__ == "__main__":
    main()

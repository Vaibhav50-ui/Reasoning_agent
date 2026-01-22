import re
import ollama
from ddgs import DDGS


# TOOL 1: SEARCH
def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if results:
            return "\n".join([f"- {r.get('body','')} (source: {r.get('href','')})" for r in results])

        return "No results found."
    except Exception as e:
        return f"Error during search: {e}"
    
# TOOL 2: CALCULATE
def calculate(expression: str):
    # allow only safe math
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s\^]+", expression):
        return {"status": "error", "error": "Invalid math expression"}

    try:
        # Support ^ like exponent
        expression = expression.replace("^", "**")
        val = eval(expression, {"__builtins__": {}})
        return {"status": "success", "value": float(val)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    

def run_agent(question):
    print(f"\n--- Question: {question} ---")

    system_prompt = (
        "You are a reasoning agent.\n"
        "You have 2 tools: SEARCH and CALCULATE.\n\n"
        "FORMAT:\n"
        "Thought: ...\n"
        "ACTION: SEARCH: <query>\n"
        "OR\n"
        "ACTION: CALCULATE: <expression>\n"
        "OR\n"
        "ANSWER: <final answer>\n\n"
        "Rules:\n"
        "- After you receive a successful calculation result, you MUST give ANSWER.\n"
        "- Do not repeat calculations unnecessarily.\n"
    )

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    max_steps = 12
    calc_count = 0
    search_count = 0
    last_calc_value = None

    for i in range(max_steps):
        response = ollama.chat(model="llama3.2:latest", messages=history)
        content = response["message"]["content"].strip()
        history.append({"role": "assistant", "content": content})

        # ----------------------------
        # ACTION: SEARCH
        # ----------------------------
        if "ACTION: SEARCH:" in content:
            search_count += 1
            query = content.split("ACTION: SEARCH:")[1].split("\n")[0].strip()
            print(f"Step {i+1}: Searching web for '{query}'...")
            result = search_web(query)

            history.append({"role": "user", "content": f"OBSERVATION: {result}"})

        # ----------------------------
        # ACTION: CALCULATE
        # ----------------------------
        elif "ACTION: CALCULATE:" in content:
            calc_count += 1
            expr = content.split("ACTION: CALCULATE:")[1].split("\n")[0].strip()
            print(f"Step {i+1}: Calculating '{expr}'...")
            result = calculate(expr)

            history.append({"role": "user", "content": f"OBSERVATION: {result}"})

            # If calc worked, force final answer
            if result["status"] == "success":
                last_calc_value = result["value"]
                history.append({
                    "role": "user",
                    "content": f"OBSERVATION: Calculation succeeded with value {last_calc_value}. "
                               f"Now you MUST provide ANSWER using this value."
                })

            # prevent endless calculations
            if calc_count >= 4:
                history.append({
                    "role": "user",
                    "content": "OBSERVATION: You have made enough calculations. Provide ANSWER now."
                })

        # ----------------------------
        # ANSWER
        # ----------------------------
        elif "ANSWER:" in content:
            final_answer = content.split("ANSWER:")[1].strip()
            print(f"\n Final Answer: {final_answer}")
            return final_answer

        else:
            print(f"Step {i+1} (Thinking): {content}")

    # fallback finalization
    if last_calc_value is not None:
        fallback = f"Approx final value: {last_calc_value:.2f}"
        print("\n Fallback Answer:", fallback)
        return fallback

    print("\n Agent failed to answer within step limit.")
    return None

if __name__ == "__main__":
    run_agent("How many liters of petrol can I buy in Delhi today with ₹1500?")
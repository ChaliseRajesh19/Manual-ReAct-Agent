import requests

API = "https://manual-react-agent.onrender.com/ask"


def ask_exam_buddy(question, session_id):
    payload = {"question": question, "session_id": session_id}

    try:
        response = requests.post(API, json=payload, timeout=120)

        print("STATUS:", response.status_code)
        print("BODY:", response.text)

        response.raise_for_status()

        return response.json()["answer"]

    except Exception as e:
        return f"Error: {e}"

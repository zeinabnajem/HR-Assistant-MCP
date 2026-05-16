from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

from data import employees, leave_balance
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

mcp = FastMCP("HR Assistant")

# Optional LLM (only used for interview questions)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


@mcp.tool()
def get_employee_details(employee_id: str) -> dict:
    """Get employee details (name, department, role) using employee ID."""
    if employee_id in employees:
        return {
            "employee_id": employee_id,
            **employees[employee_id]
        }
    return {"error": f"Employee with ID {employee_id} not found"}


@mcp.tool()
def check_leave_balance(employee_id: str) -> dict:
    """Check remaining leave balance for an employee using their ID."""
    if employee_id in leave_balance:
        return {
            "employee_id": employee_id,
            "remaining_leave_days": leave_balance[employee_id]
        }
    return {"error": f"Leave balance not found for employee ID {employee_id}"}


@mcp.tool()
def generate_interview_questions(job_role: str) -> dict:
    """Generate 5 interview questions for a given job role using AI."""

    prompt = f"""
    You are an HR expert.

    Generate 5 interview questions for the role: {job_role}

    Return only the questions as a numbered list.
    """

    response = llm.invoke(prompt)

    return {
        "job_role": job_role,
        "questions": response.content
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
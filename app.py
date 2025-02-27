__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import yaml
import warnings
from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from typing import List
from helper import load_env
import pandas as pd

from crewai_tools import FileReadTool
csv_tool = FileReadTool(file_path='./transactions.csv')

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_env()

# Set OpenAI model (Update with your model choice)
os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'

# Load configurations from YAML files
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

agents_config = load_yaml('config/agents.yaml')
tasks_config = load_yaml('config/tasks.yaml')

financial_analysis_agent = Agent(config=agents_config['financial_analysis_agent'],tools=[csv_tool])
budget_planning_agent = Agent(config=agents_config['budget_planning_agent'],tools=[csv_tool])
financial_viz_agent = Agent(config=agents_config['financial_viz_agent'],allow_code_execution=False)

suggestion_generation_agent = Agent(
  config=agents_config['suggestion_generation_agent'],
  tools=[csv_tool]
)

# Create tasks
expense_analysis = Task(
  config=tasks_config['expense_analysis'],
  agent=financial_analysis_agent
)

budget_management = Task(
  config=tasks_config['budget_management'],
  agent=budget_planning_agent
)

financial_visualization = Task(
  config=tasks_config['financial_visualization'],
  agent=financial_viz_agent
)

final_report_assembly = Task(
  config=tasks_config['final_report_assembly'],
  agent=budget_planning_agent,
  context=[expense_analysis, budget_management, financial_visualization]
)


#Create crew
finance_crew = Crew(
  agents=[
    financial_analysis_agent,
    budget_planning_agent,
    financial_viz_agent
  ],
  tasks=[
    expense_analysis,
    budget_management,
    financial_visualization,
    final_report_assembly
  ],
  verbose=True
)

# Streamlit UI
st.set_page_config(page_title="AI Project Planner", layout="wide")
st.title("🚀 Personal Financial Adviser")
st.write("⚡Effortless managing monet with AI blah blah ⚡")

if st.button("Generate"):
    result = finance_crew.kickoff()
    st.write(result.raw)
    

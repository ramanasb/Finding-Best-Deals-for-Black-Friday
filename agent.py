
from google.adk.agents.llm_agent import LlmAgent, Agent
from google.adk.tools import google_search
from pydantic import BaseModel, Field
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse, LlmRequest
from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.sequential_agent import SequentialAgent

# --- Define Output Schema ---
class DealRecommendation(BaseModel):
    subject: str = Field(
        description="The subject line of the email. Should be concise and descriptive."
    )
    body: str = Field(
        description="The main content of the email. Should be well-formatted and contain deal recommendations."
    )
# --- Create Search Agents ---

root_output_agent = LlmAgent(
    model='gemini-3-pro-preview',
    name='root_output_agent',
    description='best google search agent.',
    instruction="""
    You are a helpful assistant that can use the following tools:
    - google_search: Use this tool to search the web for relevant information.
    When you receive a user query, display only summary of the search results with links. """,
   tools=[google_search],
   #after_model_callback=refine_after_model_modifier,
   output_key="search_summary"
)
#--- Create Summary Agent ---

summary_agent = LlmAgent(
    model='gemini-3-pro-preview',
    name='summary_agent',
    description='best google search agent.',
    instruction="""
    you are an expert deal recommendation agent.
    You are given a search summary from the previous agent.
    Based on the search summary, provide a concise deal recommendation.
    recommendation should be in the format:
    **Recommendation**: <your recommendation include buillet points and provide links if any>
    """,
   #tools=[google_search],
   #before_model_callback=log_sumllm_request,
   output_key="final_recommendation"
  )


# --- Create Email Generator Agent ---
email_agent = LlmAgent(
    name="email_agent",
    model="gemini-3-pro-preview",
    instruction="""
        You are an Email Generation Assistant.
       generate email with deal recommendations.

        GUIDELINES:
        - Create an appropriate subject line (concise and relevant)
        - Write a well-structured email body with:
            * Informal greeting
            * Clear and concise with deal recommendations
            * informative to make decision
            * agent name as signature
        - Suggest relevant attachments if applicable (empty list if none needed)
        - Email tone should match the purpose (formal for business, friendly for colleagues,  makreting tone)
        - Keep emails concise but complete

        
    """,
    description="Generates deal recommendation emails with structured subject and body",
    output_schema=DealRecommendation,
    output_key="email",
)

deal_recommendation_agent = SequentialAgent(
    name="deal_recommendation_agent",
    sub_agents=[root_output_agent, summary_agent, email_agent],
    description="Deal recommendation agent that uses search and summary agents, and email generation.",
    # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
)
root_agent = deal_recommendation_agent
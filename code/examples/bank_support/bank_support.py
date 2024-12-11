"""
An example agent for Bank Support.
"""
import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Optional

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from utils.llm_model import LLMModel

logfire.configure()


class DatabaseConn:
    """
    This is a fake database class. In reality, it would connect to the database
    to fetch data.
    """

    @classmethod
    async def customer_name(cls, *, customer_id: int) -> str | None:
        """
        Returns the customer name based on the ID.
        """
        if customer_id == 123:
            return 'Sid Saha'

        return None

    @classmethod
    async def customer_currency(cls, *, customer_id: int) -> str | None:
        """
        Returns the customer's currency based on the customer ID.
        """
        if customer_id == 123:
            return 'Rs.'

        return None

    @classmethod
    async def customer_balance(cls, *, customer_id: int, include_pending: bool) -> float:  # pylint: disable=unused-argument
        """
        Returns the account balance of the customer based on the ID.
        """
        if customer_id == 123:
            return 123.45

        raise ValueError('Customer not found')


@dataclass
class SupportDependencies:
    """
    The dependencies needed by ths agent.
    """
    customer_id: int
    db: DatabaseConn


class SupportResult(BaseModel):
    """
    The data model produced by this agent.
    """
    support_advice: str = Field(
        description='The support advice returned to the customer.',
        example='This is the advice returned by the agent.',
    )
    block_card: Optional[bool] = Field(
        description='Whether to block their card',
        example=False,
    )
    risk: Optional[int] = Field(
        description='Risk level of the query', ge=0, le=10,
        example=2,
    )


def generate_json(model: BaseModel) -> str:
    """
    Generates a JSON schema example to pass to the LLM.
    """
    schema = model.model_json_schema()
    template = {}

    for prop, details in schema.get('properties', {}).items():
        eg = details.get('example')
        template[prop] = eg

    return json.dumps(template)


# create the agent
m = LLMModel()

# for ollama
# m.model_type = 'ollama'
# m.ollama_model_name = 'llama3.1:8b'

# for groq
m.model_type = 'groq'
m.groq_model_name = 'llama-3.3-70b-versatile'

llm_model = m.fetch_model()

support_agent = Agent(
    llm_model,
    deps_type=SupportDependencies,
    # result_type=SupportResult,  # NOTE: this does not work, prompting works better.
    system_prompt=(
        'You are a support agent in our bank, provide the the customer '
        'some support advice as per their request and judge the risk level of their query. '
        "When reporting the customer's balance, report using the customer's preferred currency. "
    ),
)


@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    """
    Fetches and returns the customer name.
    """
    deps: SupportDependencies = ctx.deps
    customer_name = await deps.db.customer_name(customer_id=deps.customer_id)
    return f'The name of the customer is {customer_name}'


@support_agent.system_prompt
async def return_json() -> str:
    """
    Constructs the return type that the LLM should return.
    """
    prompt = (
        f'Please return ONLY a JSON string like this example:  {generate_json(SupportResult)}. '
        'If any field in the JSON is not applicable, return its value as `null`. '
        "Make sure to include the customer's name in the final support_advice field. "
    )
    return prompt


@support_agent.tool
async def customer_balance(ctx: RunContext[SupportDependencies], include_pending: bool) -> str:
    """
    Returns the customer's current account balance.
    """
    deps: SupportDependencies = ctx.deps
    balance = await deps.db.customer_balance(
        customer_id=deps.customer_id, include_pending=include_pending,
    )
    return f'{balance:.2f}'


@support_agent.tool
async def customer_preffered_currency(ctx: RunContext[SupportDependencies]) -> str:
    """
    Return's the customer's currency.
    """
    deps: SupportDependencies = ctx.deps
    currency = await deps.db.customer_currency(customer_id=deps.customer_id)
    print(f'======= Currency: {currency} ======')
    return f'{currency}'


async def main(input_text: str, customer_id: int) -> None:
    """
    The main function to execute.
    """
    deps = SupportDependencies(
        customer_id=customer_id, db=DatabaseConn(),
    )
    result = await support_agent.run(input_text, deps=deps)

    res = SupportResult.model_validate_json(result.data)
    print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_text', required=True, type=str,
    )
    parser.add_argument(
        '--customer_id', required=True, type=int,
    )
    args = parser.parse_args()

    asyncio.run(
        main(args.input_text, args.customer_id),
    )

#!/usr/bin/env python3
"""Test gpt-5-nano with legal question."""

import asyncio
import subprocess


async def test():
    from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
    from openai import AsyncAzureOpenAI
    
    # Get config
    result = subprocess.run(['azd', 'env', 'get-values'], capture_output=True, text=True)
    env = {}
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, _, value = line.partition('=')
            env[key] = value.strip('"')
    
    endpoint = env.get('AZURE_OPENAI_ENDPOINT', '')
    deployment = env.get('AZURE_OPENAI_CHATGPT_DEPLOYMENT', 'gpt-4o')
    
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, 'https://cognitiveservices.azure.com/.default')
    
    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version='2024-12-01-preview',
        azure_ad_token_provider=token_provider,
    )
    
    print(f'Testing {deployment}...\n')
    
    # Test 1: Simple question
    print('Test 1: Simple question (no reasoning needed)')
    response = await client.chat.completions.create(
        model=deployment,
        messages=[{'role': 'user', 'content': 'What is CPR Part 44 about? Answer in 10 words.'}],
        max_completion_tokens=50,
    )
    print(f'Response: {response.choices[0].message.content}')
    print(f'Tokens - reasoning: {response.usage.completion_tokens_details.reasoning_tokens}, completion: {response.usage.completion_tokens}\n')
    
    # Test 2: With context
    print('Test 2: With legal context')
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {'role': 'system', 'content': 'You are a legal assistant. Answer briefly and cite CPR rules.'},
            {'role': 'user', 'content': 'What are the general rules about costs? Include a CPR citation.'}
        ],
        max_completion_tokens=200,
    )
    print(f'Response: {response.choices[0].message.content}')
    print(f'Tokens - reasoning: {response.usage.completion_tokens_details.reasoning_tokens}, completion: {response.usage.completion_tokens}\n')
    
    # Test 3: Higher token limit
    print('Test 3: Complex legal query with 5000 tokens')
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {'role': 'system', 'content': 'You are a legal assistant specializing in UK CPR.'},
            {'role': 'user', 'content': 'Explain the general rules about costs under CPR Part 44. Include specific rule citations.'}
        ],
        max_completion_tokens=5000,
    )
    print(f'Response length: {len(response.choices[0].message.content or "")} chars')
    print(f'Response: {(response.choices[0].message.content or "")[:200]}...')
    print(f'Tokens - reasoning: {response.usage.completion_tokens_details.reasoning_tokens}, completion: {response.usage.completion_tokens}')
    
    await credential.close()


if __name__ == "__main__":
    asyncio.run(test())

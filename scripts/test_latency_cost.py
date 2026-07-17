"""Quick latency + cost comparison: gpt-5.4-mini vs gpt-5.4 full."""
import time
from azure.identity import AzureDeveloperCliCredential
from openai import AzureOpenAI

ENDPOINT = "https://cog-gz2m4s637t5me-us2.openai.azure.com/"
API_VERSION = "2024-12-01-preview"
TENANT_ID = "3bfe16b2-5fcc-4565-b1f1-15271d20fecf"

cred = AzureDeveloperCliCredential(tenant_id=TENANT_ID)
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION,
    azure_ad_token_provider=lambda: cred.get_token("https://cognitiveservices.azure.com/.default").token,
)

SOURCES = """[1]: (Category: Civil Procedure Rules | Source: Part 24): 24.2 The court may give summary judgment against a claimant or defendant if it considers that the party has no real prospect of succeeding and there is no other compelling reason for trial.
[2]: (Category: Civil Procedure Rules | Source: Part 24): 24.3 The court may fix a hearing on its own initiative or on application. Respondent must file written evidence at least 7 days before.
[3]: (Category: Civil Procedure Rules | Source: Part 3): 3.4(2) The court may strike out a statement of case disclosing no reasonable grounds, abuse of process, or failure to comply with rules.
[4]: (Category: Commercial Court Guide | Source: Case management): Applications for summary judgment should be made promptly. Skeleton must identify the CPR 24.2 test.
[5]: (Category: Civil Procedure Rules | Source: Part 24): 24.4 A claimant may not apply for summary judgment until defendant has filed acknowledgment or defence."""

SYSTEM = "Assistant helps with English civil court procedure. Be brief. Cover 3-4 key aspects. Answer ONLY from sources. Each sentence ends with one citation. Possible citations: [1] [2] [3] [4] [5]\n"

QUESTIONS = [
    "How do I get a court to decide my case quickly without a full trial?",
    "What documents do I have to share with the other side in a lawsuit?",
    "How do I apply for an injunction to stop someone doing something urgently?",
    "How do I serve court documents on the other party?",
    "How do I challenge an arbitration award in the Commercial Court?",
    "How do I bring a trust dispute to the Chancery Division?",
]

print(f"{'Q#':<4} {'mini_ms':>8} {'full_ms':>8} {'ratio':>6} | {'m_in':>6} {'m_out':>6} {'f_in':>6} {'f_out':>6}")
print("-" * 70)

mini_times, full_times = [], []
mini_in_tot, mini_out_tot, full_in_tot, full_out_tot = [], [], [], []

for i, q in enumerate(QUESTIONS):
    user_msg = q + "\n\nSources:\n\n" + SOURCES
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]

    t0 = time.perf_counter()
    r_mini = client.chat.completions.create(model="gpt-5.4-mini", messages=msgs, max_completion_tokens=1024, temperature=0.3)
    mini_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r_full = client.chat.completions.create(model="gpt-5.4", messages=msgs, max_completion_tokens=1024, temperature=0.3)
    full_ms = (time.perf_counter() - t0) * 1000

    ratio = full_ms / mini_ms if mini_ms > 0 else 0
    mini_times.append(mini_ms)
    full_times.append(full_ms)

    m_in = r_mini.usage.prompt_tokens if r_mini.usage else 0
    m_out = r_mini.usage.completion_tokens if r_mini.usage else 0
    f_in = r_full.usage.prompt_tokens if r_full.usage else 0
    f_out = r_full.usage.completion_tokens if r_full.usage else 0
    mini_in_tot.append(m_in); mini_out_tot.append(m_out)
    full_in_tot.append(f_in); full_out_tot.append(f_out)

    print(f"  {i+1:<4} {mini_ms:>7.0f}  {full_ms:>7.0f}  {ratio:>5.1f}x | {m_in:>6} {m_out:>6} {f_in:>6} {f_out:>6}")

n = len(mini_times)
avg_mini = sum(mini_times) / n
avg_full = sum(full_times) / n
print("-" * 70)

print(f"\n  LATENCY (answer generation only)")
print(f"    mini avg:  {avg_mini:,.0f} ms")
print(f"    full avg:  {avg_full:,.0f} ms")
print(f"    ratio:     {avg_full/avg_mini:.1f}x slower")
print(f"    delta:     +{avg_full - avg_mini:,.0f} ms per request")

avg_m_in = sum(mini_in_tot) / n
avg_m_out = sum(mini_out_tot) / n
avg_f_in = sum(full_in_tot) / n
avg_f_out = sum(full_out_tot) / n
print(f"\n  TOKENS (averages per request)")
print(f"    mini:  {avg_m_in:.0f} in / {avg_m_out:.0f} out")
print(f"    full:  {avg_f_in:.0f} in / {avg_f_out:.0f} out")

# Azure GlobalStandard pricing (approximate April 2026)
# gpt-5.4-mini: $0.40/1M input, $1.60/1M output
# gpt-5.4:      $2.50/1M input, $10.00/1M output
mini_cost = (avg_m_in * 0.40 + avg_m_out * 1.60) / 1_000_000
full_cost = (avg_f_in * 2.50 + avg_f_out * 10.00) / 1_000_000

print(f"\n  COST PER REQUEST (GlobalStandard estimate)")
print(f"    mini:  ${mini_cost:.6f}")
print(f"    full:  ${full_cost:.6f}")
print(f"    ratio: {full_cost/mini_cost:.1f}x more expensive")

daily = 500
monthly_mini = mini_cost * daily * 30
monthly_full = full_cost * daily * 30
print(f"\n  MONTHLY PROJECTION (500 queries/day)")
print(f"    mini:  ${monthly_mini:.2f}/month")
print(f"    full:  ${monthly_full:.2f}/month")
print(f"    delta: +${monthly_full - monthly_mini:.2f}/month")

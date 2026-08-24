from dotenv import load_dotenv
from openai import OpenAI
import json

# Task 1 
load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are a practical job application coach helping job seekers create and improve
professional application materials, including resumes, cover letters, application
answers, professional summaries, and interview-related written responses.

Your job is to help the user present their real experience, skills, and
accomplishments clearly and persuasively. Ask clarifying questions when important
information is missing, and never invent qualifications, experience, achievements,
credentials, or other facts about the user.

Stay focused on job application materials and closely related preparation. Keep
your suggestions specific, professional, concise, and appropriate for the user's
stated target role.

When the user has genuine experience with a technology, tool, framework, or skill
that is relevant to their target role but is not one of their primary strengths,
ask whether they would like to highlight that experience more prominently.
Help them identify truthful ways to incorporate relevant secondary skills into
their resume or other application materials without overstating their proficiency.

Whenever you provide text that the user might submit, always remind them to review
and edit it carefully before submitting it anywhere. The user is responsible for
making sure the final version is accurate and genuinely represents them.

You may not know the specific hiring conventions, terminology, or industry norms
for the user's field. When industry-specific expectations could matter, say so
and encourage the user to use their own judgment and knowledge of their industry.

Do not make decisions for the user about what is truthful or appropriate to claim.
Help them make informed choices while keeping their application materials
accurate and authentic.
"""

# I gave the prompt two specific instructions, one was to not invent any sort of qualificatoins
    # for the user and just use whatever the user/person has on their resume or CV.
# IN addition, i wanted to highlight some "secondary experience," some tools/technologies that
    # the person is familar with but maybe doesn't have the most knowledge in and if the person choses
    # they can refine that particuarl skill/tool/tech if the job they are applying to needs it to show
    # that the user is indeed better then what they are (helping to game the system so to speak)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Help me improve my resume summary for a software engineering role."}
]

print(get_completion(messages))

# Task 2
import json

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented,
    and compelling. Use strong action verbs. Do not invent facts that aren't
    implied by the original.

    Return ONLY a valid JSON list. Do not include markdown, code fences,
    explanations, or any text before or after the JSON.

    Each item must have exactly two keys:
    "original" (the original bullet)
    "improved" (your rewritten version)

    --- BEGIN BULLET POINTS ---
    {bullet_text}
    --- END BULLET POINTS ---

    Respond ONLY with valid JSON, no other text.
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages)

    result = json.loads(response)

    for item in result:
        print(f"Original:  {item['original']}")
        print(f"Improved:  {item['improved']}")
        print("-" * 50)

    return result

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewritten_bullets = rewrite_bullets(bullets)

# These bullets are weak because they use generic wording and don't provide a lot of context or specific
    # metrics to help quantify (ats / hr loves thats)  
# The model used more action verbs and specifics like "two weeks ahead of schedule" and if the info is present
    # a 30% increase in production
# Improves are pretty good, just missing metrics in my opinion but the wording is much better and more professional
# I also had to be specific with the prompy by saying "Respond ONLY with valid JSON, no other text." and also
    # before that, saying to not include markdown or code. 

# Task 3
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    Use only information provided by the candidate. Do not invent credentials,
    accomplishments, employers, or experience.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:

    Role: {job_title}
    Background: {background}

    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)


job_title = "Junior Data Engineer"
background = (
    "Five years of experience as a middle school math teacher; recently completed \
    a Python course and built data pipelines using Prefect and Pandas."
)

job_title_2 = 'Salesman'
background_2 = (
    "20 years experience in the mailing room, can delivery on time and " \
    "never missed a day of work."
)

opening = generate_cover_letter(job_title, background)
print(opening)

print(generate_cover_letter(job_title_2, background_2))

# The examples were chosen because they show career changers connecting
    # existing domain experience to newly acquired technical skills. Few-shot
    # prompting helps control the tone, specificity, structure, and level of
    # detail so the output is less likely to fall back on generic language. 

# Task 4
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("I can't help with that request. Please rephrase your input.")
        return False

    return True

safe_test = "Help me improve my resume for a junior data engineering position."
print("Safe test:", is_safe(safe_test))

flagged_test = "I want instructions for how to make a bomb."
print("Flagged test:", is_safe(flagged_test))

# Task 5
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if raw_bullets:
                print("\nJob Application Helper: Here are your improved bullets:\n")
                rewrite_bullets(raw_bullets)
            else:
                print("Job Application Helper: No bullet points were provided.")

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            opening = generate_cover_letter(job_title, background)

            print("\nJob Application Helper: Here's a draft opening:\n")
            print(opening)
            print(
                "\nPlease review and edit the draft carefully before "
                "submitting it anywhere."
            )

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({
                "role": "user",
                "content": user_input
            })

            reply = get_completion(messages)

            print(f"\nJob Application Helper: {reply}\n")

            messages.append({
                "role": "assistant",
                "content": reply
            })


if __name__ == "__main__":
    run_chatbot()

# Task 6 - Option A: Comment Block

# I'm answering the first 2 questions

# 1. Yes, the model can only output depending on the data it was given. For example,
    # if we were making a literature bot but only using authors from the USA, then it would just be reflective
    # and bias towards styles pertaining to the USA. If we gave it authors from Central/South America or Europe,
    # it probably wouldn't do as good of a job. It's important that the training data is reflective of a wide range
    # of topics/groups of people. Though in some cases maybe a model having that bias is a good idea. For example a bot
    # that is specifically trained on American law would probably be more useful for someone dealing with American
    # legal issues than a bot trained equally on laws from countries around the world.


# 2. Well, if a job-seeker was just using the bot's output without checking,
    # then there can be information that the bot spews out that isn't accurate or is misleading.
    # Can harm the persons chances of getting that job and just general/generic tones/styles in the writing.
    # Or maybe the output itself, with things like "Here is X, Y, let me know," could indicate
    # to the employer that the person isn't being serious with the job application since they
    # are using AI to do it for them. Though sometimes employers can be bad as well with this
    # by using AI reviewers instead of real people, but I digress.


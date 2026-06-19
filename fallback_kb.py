import re

_GREETING = re.compile(
    r"^(hi|hey|hello|howdy|good\s+(morning|afternoon|evening)|greetings)\b",
    re.IGNORECASE,
)
_GOODBYE = re.compile(
    r"\b(bye|goodbye|see\s+you|farewell|take\s+care)\b",
    re.IGNORECASE,
)
_ACK = re.compile(
    r"^(ok|okay|cool|alright|sure|thanks|thank\s+you)[\s!.?]*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "Hey, I'm IVY, your official AI assistant from Internetworks. "
    "I'm here to help you with anything related to our company, services, or team. "
    "How can I assist you today?"
)
_GOODBYE_RESPONSE = (
    "Thank you for talking to me. "
    "If you need my assistance in the future, I'd be happy to help!"
)
_ACK_RESPONSE = (
    "Is there anything else I can help you with regarding Internetworks?"
)
_DEFAULT_RESPONSE = (
    "I'm sorry, I can only answer questions related to Internetworks "
    "based on the information I have."
)

KB_ENTRIES = [
    {
        "keywords":[
            "contact",
            "connect",
            "phone",
        ],
        "response":(
            "Contact Info:\n"
            "muizznaveed@internetworks.io\n"        
            "huzaifaarif@internetworks.io\n"
            "+92 309 9889911\n"
            "Pakistan - 4A R Block Rd, Johar Town,Phase 2, Block R Lahore"
        ),
    },
    {
        "keywords": [
            "founder",
            "muizz",
            "naveed",
            "who started",
            "who created",
        ],
        "response": (
            "Muizz Naveed Ali is the Founder of Internetworks. "
            "He leads company vision, strategy, and service expansion."
        ),
    },
    {
        "keywords": [
            "director",
            "huzaifa",
            "muhammad huzaifa",
            "operations",
        ],
        "response": (
            "Muhammad Huzaifa Arif is the Director of Internetworks. "
            "He oversees operations, partnerships, and project execution."
        ),
    },
    {
        "keywords": [
            "usama",
            "senior software",
            "ai integration",
            "chatbot engineer",
        ],
        "response": (
            "Usama Javed is a Senior Software Engineer (AI Integration) at Internetworks. "
            "He develops and integrates AI technologies such as chatbots and "
            "automated learning management systems (ALMS)."
        ),
    },
    {
        "keywords": [
            "danial",
            "frontend",
            "ui",
            "ux",
        ],
        "response": (
            "Danial Ayyaz is a Frontend Developer at Internetworks. "
            "He designs and builds user-friendly interfaces for web and mobile applications."
        ),
    },
    {
        "keywords": [
            "soha",
            "prompt engineer",
            "prompt engineering",
        ],
        "response": (
            "Soha Sarwar is an AI Prompt Engineer at Internetworks. "
            "She crafts, tests, and optimizes prompts for AI-driven solutions."
        ),
    },
    {
        "keywords": [
            "team",
            "employees",
            "staff",
            "who works",
            "people at",
        ],
        "response": (
            "The Internetworks team includes Muizz Naveed Ali (Founder), "
            "Muhammad Huzaifa Arif (Director), Usama Javed (Senior Software Engineer, "
            "AI Integration), Danial Ayyaz (Frontend Developer), and "
            "Soha Sarwar (AI Prompt Engineer)."
        ),
    },
    {
        "keywords": [
            "mission",
            "purpose",
            "why internetworks",
            "what we believe",
        ],
        "response": (
            "At Internetworks our mission is to empower businesses by delivering reliable, "
            "scalable, and future-ready IT solutions. We drive growth through innovation, "
            "seamless technology integration, and customer-first service."
        ),
    },
    {
        "keywords": [
            "founded",
            "when started",
            "established",
            "2020",
            "history",
        ],
        "response": (
            "Internetworks was founded in 2020 and is based in Massachusetts, "
            "United States, in the Greater Boston area."
        ),
    },
    {
        "keywords": [
            "location",
            "where are you",
            "where is internetworks",
            "boston",
            "massachusetts",
            "address",
            "based",
        ],
        "response": (
            "Internetworks is based in Massachusetts, United States, "
            "in the Greater Boston area."
        ),
    },
    {
        "keywords": [
            "service",
            "services",
            "what do you do",
            "what do you offer",
            "what does internetworks",
            "offerings",
        ],
        "response": (
            "Internetworks offers Artificial Intelligence Solutions, Software Development, "
            "Cloud & IT Services, Business Solutions, and Specialized Services "
            "including Salesforce. This includes AI as a Service, full stack development, "
            "Microsoft 365, cloud-native architectures, CRM implementations, and more."
        ),
    },
    {
        "keywords": [
            "artificial intelligence",
            " ai ",
            "machine learning",
            "ml",
            "automation",
            "ai as a service",
        ],
        "response": (
            "Internetworks provides Artificial Intelligence Solutions including "
            "AI as a Service and Prompt Engineering to deploy, integrate, and optimize "
            "AI models for business needs."
        ),
    },
    {
        "keywords": [
            "software development",
            "full stack",
            "web development",
            "mobile app",
            "saas",
        ],
        "response": (
            "Internetworks offers Software Development services including "
            "Full Stack Development for web and mobile applications and "
            "SaaS Application Development for scalable, secure platforms."
        ),
    },
    {
        "keywords": [
            "cloud",
            "microsoft 365",
            "office 365",
            "api integration",
            "cybersecurity",
            "security",
        ],
        "response": (
            "Internetworks Cloud & IT Services include Microsoft 365 setup and management, "
            "cloud-native architectures, API integrations, data-driven automation, "
            "and enterprise-grade security."
        ),
    },
    {
        "keywords": [
            "crm",
            "salesforce",
            "staff augmentation",
            "outsourcing",
            "support",
            "maintenance",
        ],
        "response": (
            "Internetworks Business Solutions include CRM implementations, "
            "performance optimization, staff augmentation, continued long-term support, "
            "and specialized Salesforce services."
        ),
    },
    {
        "keywords": [
            "ivy",
            "who are you",
            "your name",
            "chatbot",
            "assistant",
        ],
        "response": (
            "I'm IVY, Internetworks' official AI assistant. "
            "I'm here to help with questions about our company, services, and team."
        ),
    },
]


def _score_entry(message: str, keywords: list[str]) -> int:
    # Pad with spaces so single-word boundary checks work without regex.
    # All matching uses the padded string — never fall back to the raw
    # message, which would allow partial-word matches like "ui" in "build".
    padded = f" {message} "
    return sum(1 for keyword in keywords if keyword in padded)


def get_fallback_response(user_message: str) -> str:
    """Return a rule-based answer when the LLM is unavailable."""
    normalized = user_message.strip().lower()

    if _GREETING.search(normalized):
        return _GREETING_RESPONSE
    if _GOODBYE.search(normalized):
        return _GOODBYE_RESPONSE
    if _ACK.match(normalized):
        return _ACK_RESPONSE

    best_score = 0
    best_response = None
    for entry in KB_ENTRIES:
        score = _score_entry(normalized, entry["keywords"])
        if score > best_score:
            best_score = score
            best_response = entry["response"]

    if best_response:
        return best_response

    return _DEFAULT_RESPONSE

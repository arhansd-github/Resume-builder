import asyncio
import logging

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Import the graph builder functions (use absolute import so script can be run directly)
from graph_builder import build_resume_graph, run_interactive_session
from agents.resume_builder_state import ResumeBuilderState

async def main():
    """Main function to build and run the resume builder graph."""
    # Pre-populate initial state with provided resume + JD info
    initial_state = ResumeBuilderState()
    # Raw resume sections content (example / placeholder strings)
    initial_state.resume_sections = {
        "skills": ["java", "python", "aws", "docker", "kubernetes"],
        "experiences": "Senior Developer at XYZ Corp building data pipelines and APIs. Led a team of 5 developers.",
        "education": "B.Tech Computer Science, ABC University, 2022. GPA: 3.8/4.0",
        "projects": [
            "Real-time analytics platform (Python, Kafka, Redis)",
            "Resume intelligence tool (LLM, embeddings, vector DB)",
        ],
        "summary": "Results-driven software engineer with 5+ years of experience in building scalable applications. Specialized in Python, cloud technologies, and system design.",
        "contact": {
            "email": "john.doe@example.com",
            "phone": "+1 (555) 123-4567",
            "linkedin": "linkedin.com/in/johndoe",
            "github": "github.com/johndoe"
        },
        "certificates": [
            "AWS Certified Solutions Architect - Associate (2023)",
            "Google Cloud Professional Data Engineer (2022)",
            "Docker Certified Associate (2021)"
        ],
        "publications": [
            "Optimizing Microservices Communication: A Case Study (2023)",
            "Machine Learning Model Deployment Best Practices (2022)"
        ],
        "languages": [
            {"name": "English", "proficiency": "Native"},
            {"name": "Spanish", "proficiency": "Professional Working"},
            {"name": "Hindi", "proficiency": "Native"}
        ],
        "recommendations": [
            {
                "name": "Jane Smith",
                "position": "CTO at XYZ Corp",
                "text": "John consistently delivered high-quality code and mentored junior team members.",
                "date": "2024-01-15"
            },
            {
                "name": "Mike Johnson",
                "position": "Senior Developer at ABC Tech",
                "text": "Exceptional problem-solving skills and a great team player.",
                "date": "2023-11-05"
            }
        ],
        "custom": {
            "achievements": [
                "Speaker at TechConf 2023: 'Modern Cloud Architectures'",
                "Open Source Contributor: Contributed to 5+ major projects"
            ],
            "volunteer": [
                "Mentor at Code for Good (2022-Present)",
                "Organizer at Local Hackathon (2021, 2022)"
            ]
        }
    }
    initial_state.jd_summary = ("""
        Job Title: Senior Python Developer (Cloud & Data Engineering Focus)
        Location: Remote / Hybrid
        Employment Type: Full-Time
        About the Role
        We are seeking a results-driven Python Developer with strong experience in building scalable applications, APIs, and modern cloud-based data platforms. The ideal candidate will have expertise in Python, cloud technologies (AWS/GCP), and containerization tools (Docker, Kubernetes), while also demonstrating proven impact through measurable results in past projects.
        Key Responsibilities
        Design, build, and maintain data pipelines and APIs with high reliability and performance.
        Develop real-time analytics platforms and contribute to NLP/vector database-based solutions.
        Implement infrastructure as code (Terraform) and manage scalable cloud environments (AWS, GCP, Azure preferred).
        Collaborate with cross-functional teams, ensuring CI/CD pipelines and modern DevOps practices are followed.
        Quantify and communicate the impact of delivered projects (e.g., performance improvements, user adoption, revenue growth).
        Mentor junior engineers and contribute to open-source/community initiatives.
        Required Skills & Qualifications
        Bachelor’s degree in Computer Science or related field (Master’s degree preferred).
        5+ years of experience in software engineering with strong expertise in:
        Python, Java
        AWS and GCP (Azure certification a plus)
        Docker, Kubernetes, Terraform
        CI/CD pipelines, system design, microservices
        Experience with NLP, embeddings, and vector databases.
        Strong written/oral communication, with ability to document and publish technical findings (conference papers, case studies).
        Multilingual ability (English, Spanish, Hindi preferred).
        Nice to Have
        Recent cloud certifications (AWS, GCP, Azure) and other relevant credentials.
        Strong professional network with diverse recommendations (technical leads, managers, peers).
        Previous conference talks, publications, or open-source contributions.
        Active involvement in mentorship, hackathons, or community organizations.
        What We Offer
        Competitive compensation with performance-based growth opportunities.
        Chance to work on cutting-edge AI, data, and cloud projects.
        Opportunities for professional development and continuous learning.
        Collaborative and inclusive work culture."""
    )
    initial_state.section_objects = {
            "skills": {
                "section_name": "skills",
                "alignment_score": 85,
                "missing_requirements": ["GCP", "Terraform"],
                "recommended_questions": [
                    "Do you have infrastructure as code experience?",
                    "Can you highlight any hands-on GCP projects?",
                    "Have you used Terraform in production environments?"
                ],
            },
            "experiences": {
                "section_name": "experiences",
                "alignment_score": 78,
                "missing_requirements": ["quantified impact"],
                "recommended_questions": [
                    "Can you add metrics to achievements?",
                    "Have you led cross-functional initiatives that delivered measurable results?",
                    "What revenue or efficiency impact did your projects have?"
                ],
            },
            "education": {
                "section_name": "education",
                "alignment_score": 80,
                "missing_requirements": ["Master's degree or higher"],
                "recommended_questions": [
                    "Do you have any higher qualifications?",
                    "Have you completed any postgraduate certifications or diplomas?",
                    "Are you pursuing advanced education or professional development courses?"
                ],
            },
            "projects": {
                "section_name": "projects",
                "alignment_score": 75,
                "missing_requirements": ["performance metrics", "user impact"],
                "recommended_questions": [
                    "What was the user impact of these projects?",
                    "Can you share performance improvements or efficiency gains?",
                    "Did any of these projects achieve adoption at scale?"
                ],
            },
            "summary": {
                "section_name": "summary",
                "alignment_score": 70,
                "missing_requirements": ["specific achievements", "technologies used"],
                "recommended_questions": [
                    "Can you highlight specific achievements?",
                    "Would you like to mention key technologies or tools in your summary?",
                    "Can you add a quantified impact statement in your summary?"
                ],
            },
            "contact": {
                "section_name": "contact",
                "alignment_score": 95,
                "missing_requirements": ["linked in profile"],
                "recommended_questions": [
                    "Do you have any LinkedIn or GitHub profile?",
                    "Would you like to include a personal portfolio website?",
                ],
            },
            "certificates": {
                "section_name": "certificates",
                "alignment_score": 80,
                "missing_requirements": ["recent cloud certifications"],
                "recommended_questions": [
                    "Have you considered getting certified in Azure?",
                    "Are you planning to renew or update your current certifications?",
                    "Do you have any specialized certifications in DevOps or security?"
                ],
            },
            "publications": {
                "section_name": "publications",
                "alignment_score": 60,
                "missing_requirements": ["citations", "conference details"],
                "recommended_questions": [
                    "Can you add citation details in the field of data analytics?",
                    "Were any of your works published in peer-reviewed journals or conferences?",
                    "Do you have links to public repositories or case studies for your publications?"
                ],
            },
            "languages": {
                "section_name": "languages",
                "alignment_score": 100,
                "missing_requirements": [],
                "recommended_questions": [],
            },
            "recommendations": {
                "section_name": "recommendations",
                "alignment_score": 65,
                "missing_requirements": ["diverse recommenders", "specific skills mentioned"],
                "recommended_questions": [
                    "Can you get a recommendation from a technical lead?",
                    "Would you like to include feedback highlighting specific technical skills?",
                    "Do you have recommendations from clients or cross-team collaborators?"
                ],
            },
            "custom": {
                "section_name": "custom",
                "alignment_score": 50,
                "missing_requirements": ["professional development", "community involvement details"],
                "recommended_questions": [
                    "Would you like to add any professional development activities?",
                    "Can you provide more details on your community or volunteer involvement?",
                    "Would you like to highlight any leadership roles in events or organizations?"
                ],
            }
        }

    compiled_graph = build_resume_graph(initial_state=initial_state)

    # Run the interactive session
    await run_interactive_session(compiled_graph)

if __name__ == "__main__":
    asyncio.run(main())
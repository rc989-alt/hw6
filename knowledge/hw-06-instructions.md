# HW 06: Using LLMs to Get Stuff Done

**Due:** November 19, 2025 at 11:59pm ET

## Overview
This assignment has three main components focusing on practical LLM applications: text classification, performance evaluation, and chatbot development.

## Part 1: Labeling Legislative Documents

### Exercise 1: Generate LLM Classifications
Test 9 model/prompt combinations (3 models × 3 prompt styles) on U.S. Congressional bill descriptions to classify them into 20 policy categories from the Comparative Agendas Project.

**Models to Test:**
- GPT 4.1
- GPT 5-nano
- GPT 5

**Prompt Styles:**
1. **Naive/Simple** - Basic classification request (<100 words, excluding bill text)
2. **Explicit Values** - Include full list of policy category numbers and labels
3. **Detailed/Reasoning** - Engineer an optimized prompt using best practices for chain-of-thought reasoning

**Implementation Requirements:**
- Use a standalone script for predictions (not within the Quarto document)
- Implement `chat_structured()` method for consistent outputs
- Test on small samples first (5-10 bills) before running on full dataset
- Use batch processing API to minimize costs (2x cheaper than synchronous)
- Track token usage and expenses for each combination
- Export results with all predictions and true labels

**Dataset:**
- 500 U.S. legislative bill descriptions
- 20 policy categories from Comparative Agendas Project
- Categories include: Macroeconomics, Civil rights, Health, Agriculture, Labor, Education, Environment, Energy, Immigration, Transportation, Law/crime, Social welfare, Housing, Banking/finance, Defense, Technology, Foreign trade, International affairs, Government operations, Public lands

### Exercise 2: Evaluate Performance
Compare all 9 model/prompt combinations using these metrics:
- **Accuracy**: Overall correct classification rate
- **F-measure**: Harmonic mean of precision and recall
- **Sensitivity**: True positive rate (recall)
- **Specificity**: True negative rate

Present findings in professional tables and/or figures. Discuss:
- Which combinations performed best and why?
- What patterns emerge across models and prompts?
- Which combinations offer best value relative to cost?
- Cost-effectiveness analysis (performance per dollar)

## Part 2: INFO 4940/5940 Tutor Chatbot

### Exercise 3: Build & Deploy Chatbot
Create an intelligent tutoring assistant using the shinychat framework with these capabilities:

**Functional Requirements:**
The chatbot must be able to:
- Answer questions about course topics covered in lectures and readings
- Provide guidance on assignments and projects
- Answer questions about course policies from the syllabus
- Offer study tips and resources
- Help students implement the coding techniques required for assignments

**Technical Requirements:**
- Built with shinychat framework (R or Python)
- System prompt clearly defining the role as INFO 4940/5940 tutor
- Clear instructions on what the chatbot should and should not do
- One or more RAG (Retrieval-Augmented Generation) knowledge stores containing:
  - Course syllabus
  - Lecture notes
  - Assignment instructions
  - Relevant textbooks or articles
- Deployed on Posit Connect Cloud (free tier)
- Public link visible in hw-06.qmd submission

**Enhancement Opportunities:**
Feel free to go above and beyond with:
- UI customizations for better engagement
- Additional features (code execution, visualization, examples)
- Enhanced functionality (conversation history, personalization)
- Multiple specialized knowledge stores
- Integration with course-specific tools

**Deployment Best Practices:**
- Store all required resources (data files, images, CSS) with app or in subfolder
- Create deployment-specific API key (don't reuse local key)
- Set API key as environment variable in Posit Connect Cloud deployment settings
- Test thoroughly before deploying
- Ensure all dependencies are properly specified

## Additional Requirements

**Code Quality & Workflow:**
- Set random seed for reproducibility (when applicable)
- Use caching to optimize rendering
- Make at least 3 meaningful git commits
- Update author name in document
- Label code chunks clearly and descriptively
- Format results professionally using gt (R) or great_tables (Python)

**Documentation:**
- Include code comments explaining key decisions
- Document prompt engineering choices
- Explain RAG implementation approach
- Describe chatbot design philosophy

**GAI Self-Reflection:**
Include written reflection addressing:
- How you used generative AI tools in this assignment
- What skills you acquired or strengthened
- How this assignment relates to course learning objectives
- Challenges encountered and how you overcame them

## Grading Breakdown

- Exercise 1 (LLM Classifications): 20 points
  - Correct implementation of 9 combinations
  - Proper use of batch processing
  - Accurate token tracking
  - Clean code and documentation
- Exercise 2 (Performance Evaluation): 5 points
  - All four metrics calculated correctly
  - Professional visualizations
  - Insightful analysis and discussion
- Exercise 3 (Tutor Chatbot): 25 points
  - Functional chatbot meeting all requirements
  - Effective RAG implementation
  - Quality system prompt
  - Successful deployment
  - UI/UX considerations
- **Total: 50 points**

## Submission Instructions

1. Render your hw-06.qmd document to PDF/HTML
2. Submit via Gradescope
3. Mark all assignment pages correctly in Gradescope
4. Ensure deployed chatbot link is accessible
5. Include all code files in your repository

## Tips for Success

- **Start early** - Deployment can have unexpected issues
- **Test incrementally** - Don't wait until the end to test components
- **Monitor costs** - Use batch processing and test on samples first
- **Ask questions** - Use office hours and discussion forums
- **Document thoroughly** - Future you will thank present you
- **Version control** - Commit frequently with meaningful messages

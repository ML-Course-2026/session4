# **Sample Project Specifications**

This spec document reflects the decisions made during the conversation with the LLM, focusing on the **Mental Health Support App** with dynamic dropdowns.

---

## **AI-Powered Mental Health Support Assistant (Prototype)**

This document outlines the specifications for our group project, derived from an interactive session with an AI assistant. We are a team of four beginners building an application using Gradio, Google Colab, and leveraging LLM assistance ("vibe coding"). Our goal is to create a supportive tool while emphasizing it is **not a replacement for professional medical advice or crisis intervention.**

**Chosen Application Focus:** Healthcare → Mental Health Support

---

### **Phase 1: Mock UI Demo (1 Week) — Detailed Specifications**

**Goal:**  
Create a functional Gradio user interface prototype that simulates the user interaction flow for the Mental Health Support app. This prototype will accept user selections via multiple dropdowns and display *hardcoded* mock data relevant to the selections, without actual AI processing.

---

#### **1. User Interface (Gradio):**

The interface will consist of the following components, arranged vertically:

- **Input Component 1 (Dropdown):** Labeled **"Select Main Category"**
  - Default Option: *"-- Select a mental health topic --"* (Mandatory selection required to proceed)
  - Options:  
    `["Stress & Anxiety", "Depression", "Sleep Issues", "Emotional Well-being", "General Mental Health Support"]`

- **Input Component 2 (Dropdown):** Labeled **"Select Subcategory"**
  - Default Option: *"-- Select a subtopic --"*
  - Options:  
    Initially, this might show a combined list or be disabled until the Main Category is selected.  
    **In Phase 1, the dynamic update logic is NOT required**, but the dropdown should exist. Mock logic will *pretend* to use this selection.  
    Example static list:  
    `["Coping Techniques", "Breathing Exercises", "Mindfulness Tips", "Motivational Support", "Journaling Prompts", "Professional Help Resources", "Sleep Hygiene Tips", "Relaxation Techniques", "Guided Sleep Meditation", "Self-care Activities", "Positive Affirmations", "Relationship Advice"]`

- **Input Component 3 (Dropdown):** Labeled **"Select Severity Level"**
  - Default Option: *"-- Select severity --"*
  - Options:  
    `["Mild", "Moderate", "Severe", "Crisis"]`

- **Input Component 4 (Dropdown):** Labeled **"Select Age Group"**
  - Default Option: *"-- Select age group --"*
  - Options:  
    `["Teen (13-19 years)", "Young Adult (20-30 years)", "Adult (31-50 years)", "Senior (50+ years)"]`

- **Button:** Labeled **"Get Support Tips"**

- **Output Component 1 (Text Area):** Labeled **"Your Selections"**
  - Purpose: Display the user’s selected options after clicking the button

- **Output Component 2 (Text Area):** Labeled **"Suggested Support (Mock Data)"**
  - Purpose: Show a static, pre-written response simulating helpful advice or resources

---

#### **2. Core Functionality (Mock Logic):**

When the user has made selections in all dropdowns and clicks the **"Get Support Tips"** button:

- The app reads the selected values from all four dropdowns
- Displays selections in **"Your Selections"**
- Displays a *fixed, hardcoded* message in **"Suggested Support (Mock Data)"**
- **Note:** The hardcoded response will be the same (or from a small set), regardless of selection

**Example Hardcoded Output:**
```
Based on your selections (Main Category: [User's Choice], Subcategory: [User's Choice], Severity: [User's Choice], Age Group: [User's Choice]):

Here are some general tips that might be helpful:
* Practice deep breathing exercises for 5 minutes daily.
* Consider journaling your thoughts and feelings.
* Ensure you are getting adequate sleep and nutrition.
* Reach out to a trusted friend or family member.

**Disclaimer:** This tool provides general suggestions and is not a substitute for professional medical advice, diagnosis, or treatment. If you selected 'Severe' or 'Crisis', or feel you need immediate help, please contact a crisis hotline or mental health professional immediately. [Include example hotline number/link].

(Note: This is a static example for the Phase 1 demo.)
```

---

#### **3. Technology:**

- **Language:** Python 3  
- **UI Library:** Gradio  
- **Dev Environment:** Google Colab (Free Tier)

---

#### **4. Deliverable:**

- A runnable Google Colab notebook (`.ipynb`) containing:
  - The functional Gradio mock UI with 4 dropdowns, button, and output areas

---

#### **5. Suggested Task Division (Example for 4 Members):**

- **Member 1 (UI Setup & Dropdowns 1 & 2):**  
  Set up Colab, import Gradio, implement "Main Category" and "Subcategory" dropdowns

- **Member 2 (Dropdowns 3 & 4 & Button):**  
  Add "Severity Level", "Age Group" dropdowns, and "Get Support Tips" button

- **Member 3 (Input Display & Mock Response):**  
  Write logic to read inputs and display mock data

- **Member 4 (Output Display & Integration):**  
  Implement output areas and integrate the mock display logic

---

## **Phase 2: LLM-Based Functionality (1 Week) — General Specifications**

**Goal:**  
Enhance the prototype by replacing hardcoded messages with AI-generated suggestions and making the **Subcategory** dropdown dynamically dependent on the **Main Category**.

---

#### **1. AI Integration:**

- Use either:
  - **Primary Path:** Local Hugging Face model (Colab-compatible)
  - **Fallback:** External API (e.g., Gemini)

- Handle API keys securely

---

#### **2. Dynamic UI Logic:**

- Implement Gradio logic to dynamically update **Subcategory** options based on **Main Category**

---

#### **3. Input Processing (Prompt Engineering):**

**Prompt Structure Example:**
> "Act as a supportive mental health assistant (not a therapist). Based on the user's selection of Main Category: [Category], Subcategory: [Subcategory], Severity: [Severity], and Age Group: [Age Group], provide brief, empathetic, actionable wellness tips or resources. If severity is 'Severe' or 'Crisis', strongly emphasize seeking professional help and provide crisis resource information instead of general tips. Keep the tone supportive and general."

---

#### **4. Output Handling & Safety:**

- Use prompt to call model/API  
- Display AI response  
- **If "Severe" or "Crisis" selected:**  
  - Override AI response  
  - Display predefined crisis resources + disclaimers  
- Always include disclaimer

---

#### **5. Technology:**

- Python, Gradio, Google Colab  
- Hugging Face libraries or API clients

---

#### **6. Suggested Task Division:**

- **AI Research & Backend Setup**  
- **Dynamic UI Implementation**  
- **Prompt Engineering & Integration**  
- **Output Handling & Safety**

---

## **Phase 3: Finalization (3 Days) — General Specifications**

**Goal:**  
Polish the app, highlight safety disclaimers, write documentation, and prepare final presentation.

---

#### **1. Code Refinement & Safety Check:**

- Add comments, clean code  
- Test safety logic thoroughly (Severe/Crisis handling)  
- Verify disclaimers in all outputs

---

#### **2. Documentation (`README.md`):**

Include:
- Project Goal  
- Team  
- How to Run  
- Features (dynamic dropdowns, AI integration)  
- Architecture  
- Safety Considerations  
- Challenges  
- Future Ideas

---

#### **3. Presentation Preparation:**

- Slides covering:
  - Introduction  
  - Demo (UI + AI, including safety logic)  
  - Technical Details  
  - Learnings  
  - Safety Measures

---

#### **4. Final Testing:**

- Try various input combinations  
- Focus on edge cases + dynamic behavior

---

#### **5. Deliverables:**

- Final Colab Notebook (`.ipynb`)  
- `README.md`  
- Slides

---

#### **6. Suggested Task Division:**

- **Code & Safety Lead:** Final polish + safety testing  
- **Documentation Lead:** Draft `README.md`, emphasize disclaimers  
- **Presentation Lead:** Slides + coordination  
- **Testing & Coordination Lead:** Bug checks + final testing

*(All members contribute, review safety, and practice.)*


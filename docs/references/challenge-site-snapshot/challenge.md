# Ultrastructural Segmentation of Axons and Myelin in Mouse Corpus Callosum TEM

## 1. Challenge Motivation & Biological Context

Hosted in conjunction with the **SIPAIM Conference**, this challenge addresses a major bottleneck in quantitative neurobiology: the automated and reproducible segmentation of myelinated axons and their surrounding myelin sheaths in Transmission Electron Microscopy (TEM) images.

The corpus callosum, the largest white-matter commissural tract in the mammalian brain, connects the two cerebral hemispheres through densely packed bundles of parallel myelinated axons. In the central nervous system, myelin enables rapid and energy-efficient propagation of action potentials through saltatory conduction while also providing essential metabolic support to the underlying axon. In experimental models of demyelinating and neurodegenerative disorders, including Multiple Sclerosis (MS), Amyotrophic Lateral Sclerosis (ALS), and inherited leukodystrophies, quantitative assessment of axonal morphology and myelin integrity at nanometer-scale resolution is critical for evaluating disease progression, white-matter damage, and remyelination.

Recent advances in TEM acquisition technologies have enabled the generation of large-scale, high-resolution datasets containing thousands of individual fibers. However, extracting biologically meaningful measurements from these images remains a substantial challenge. Manual annotation is extremely time-consuming, subject to observer variability, and impractical for large datasets. Although several automated approaches have been proposed, many struggle to generalize across variations in tissue preparation, staining protocols, and imaging conditions. 

This challenge focuses on **paired instance segmentation** of two topologically linked compartments for each individual fiber: the **axon (`X`)** and its surrounding **myelin sheath (`X + 1`)**. Reliable instance-level delineation of these nested structures is a prerequisite for deriving key morphometric biomarkers per fiber, including axonal diameter, myelin thickness, outer fiber diameter, and the widely used g-ratio, defined as the ratio between the inner axonal diameter and the total outer fiber diameter. By promoting the development of robust and generalizable instance segmentation methods, the challenge aims to facilitate large-scale quantitative analyses of white-matter microstructure and accelerate research on neurological disease mechanisms and therapeutic interventions.

---

## 2. How to Participate

1. **Register Your Participation (Individual or Team):** Complete the registration form on this page before **November 9, 2026 (23:59 CET)**. If you are participating as a team, check the team box and indicate your team name so we can link your registration to your final submission.
2. **Download the Training & Validation Dataset:** Access the annotated Mouse Corpus Callosum TEM dataset via the **Dataset** section.
3. **Develop Your Segmentation Pipeline:** Train and validate your instance segmentation model for axon and myelin ensuring full reproducibility.
4. **Submit Code, Method Report, and Test Predictions:** Upload your final `.zip` package through our Paris Brain Institute (ICM) OneDrive submission portal before the submission deadline.

---

## 3. Important Dates & Deadlines

| Milestone | Date | Status |
| :--- | :--- | :--- |
| **Challenge Launch & Data Release** | October 2026 | Open |
| **Registration Deadline** | November 9, 2026 (23:59 CET) | **Active** |
| **Unlabeled Test Dataset Release** | November 9, 2026 (23:59 CET) | Upcoming |
| **Final Submission Deadline** | November 13, 2026 (23:59 CET) | Upcoming |
| **Winner Announcement & SIPAIM Session** | SIPAIM Conference | Upcoming |

---

## 4.Award & Presentation Requirement

While challenge participation is open to all researchers and students worldwide, at least one member of the winning team must be registered for the **SIPAIM Main Conference (November 18–20, 2026)** or the **SIPAIM Workshop (November 16, 2026)** to officially claim the award and present their methodology during the challenge session.

---

## 5. Software Support & Participant Resources

We are thrilled to announce that **MathWorks (MATLAB)** is supporting our challenge by providing complimentary temporary software licenses to all registered participants! 
- [Local MATLAB license](https://www.mathworks.com/licensecenter/classroom/5521850/)
- [Online MATLAB license](https://www.mathworks.com/licensecenter/classroom/5521800/)

---

# Question 1: If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

I’d expand the dataset using data augmentation methods like synonym replacement, back-translation, and template-based generation. For labeling, I’d go with a semi-supervised learning approach.

# Question 2: How would you ensure your reply classifier doesn't produce biased or unsafe outputs in production?

I’d test for bias using diverse demographic text samples and apply adversarial debiasing during training. To ensure safer outputs, I’d include content filters to flag inappropriate language and set confidence thresholds to send uncertain predictions for human review. Continuous monitoring through A/B testing and feedback loops would help identify and fix issues post-deployment.

# Question 3: Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

I’d use few-shot prompting with strong email examples and include key prospect context—like industry, role, and pain points—directly in the prompt. Chain-of-thought prompting would guide the LLM to reason through personalization. I’d also add constraint-based generation to ensure prospect-specific details are included and run A/B tests on different prompt templates to optimize engagement metrics.
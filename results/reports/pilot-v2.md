# PILOT v2 (plumbing and timing only — never a result claim)

| backend | dataset | calls | failures | vector len | p50 s | p95 s | cost USD |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| jev_openrouter | agnews | 90 | 0 | [4] | 0.2224 | 0.331 | 0.001661 |
| jev_openrouter | banking77 | 90 | 0 | [72] | 0.2475 | 0.3336 | 0.008635 |
| jev_openrouter | civil_comments | 90 | 0 | [2] | 0.2536 | 0.3984 | 0.001399 |
| jev_openrouter | emotiondair | 90 | 0 | [6] | 0.2551 | 0.329 | 0.001703 |
| jev_openrouter | massive_en | 90 | 0 | [60] | 0.2499 | 0.3354 | 0.005181 |
| jev_openrouter | massive_km | 90 | 0 | [60] | 0.2493 | 0.3259 | 0.005267 |
| jev_openrouter | massive_zh | 90 | 0 | [60] | 0.2421 | 0.3249 | 0.005185 |
| jev_openrouter | sms_spam | 90 | 0 | [2] | 0.2553 | 0.3508 | 0.001345 |
| jev_openrouter | ultrafeedback_helpfulness | 90 | 0 | [5] | 0.2519 | 0.3869 | 0.00182 |
| laya_base | agnews | 30 | 0 | [4] | 0.0616 | 0.0907 | 0 |
| laya_base | civil_comments | 30 | 0 | [2] | 0.055 | 0.0728 | 0 |
| laya_base | emotiondair | 30 | 0 | [6] | 0.0611 | 0.067 | 0 |
| laya_base | sms_spam | 30 | 0 | [2] | 0.0551 | 0.06 | 0 |
| laya_base | ultrafeedback_helpfulness | 30 | 0 | [5] | 0.0787 | 0.1166 | 0 |
| laya_multilingual | massive_en | 30 | 0 | [60] | 0.0874 | 0.1003 | 0 |
| laya_multilingual | massive_km | 30 | 0 | [60] | 0.0984 | 0.1599 | 0 |
| laya_multilingual | massive_zh | 30 | 0 | [60] | 0.0961 | 0.1395 | 0 |
| qwen_logit | agnews | 30 | 0 | [4] | 0.205 | 0.3162 | 0 |
| qwen_logit | banking77 | 30 | 0 | [72] | 2.0688 | 8.1174 | 0 |
| qwen_logit | civil_comments | 30 | 0 | [2] | 0.1668 | 0.2673 | 0 |
| qwen_logit | emotiondair | 30 | 0 | [6] | 0.1959 | 0.2146 | 0 |
| qwen_logit | massive_en | 30 | 0 | [60] | 0.6698 | 1.0445 | 0 |
| qwen_logit | massive_km | 30 | 0 | [60] | 0.6976 | 1.0555 | 0 |
| qwen_logit | massive_zh | 30 | 0 | [60] | 0.6402 | 0.6899 | 0 |
| qwen_logit | sms_spam | 30 | 0 | [2] | 0.1706 | 0.1999 | 0 |
| qwen_logit | ultrafeedback_helpfulness | 30 | 0 | [5] | 0.2968 | 0.4404 | 0 |

Jev ledger: {'settled_usd': 0.032196, 'retained': 0, 'dispatches': 810}

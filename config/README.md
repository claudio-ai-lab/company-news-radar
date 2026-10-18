# Your configuration goes here

Everything in this folder except this file is git-ignored. Put your own company profile
and, if you want, your own prompt templates here:

```
config/
├── acme.yaml            # your profile (copy of examples/example-ag/config.yaml)
└── prompts/             # optional: set `prompts: {dir: prompts}` in acme.yaml
    ├── en/summarize.yaml
    └── classify.yaml    # used for any output language without its own folder
```

Run it with `newsradar run -c config/acme.yaml`.

A profile can also build on a shared base with `extends: ../examples/example-ag/config.yaml`
and override only what differs. Keep API keys in `.env` or the environment, never in YAML.

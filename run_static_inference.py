#!/usr/bin/env python3
"""Deployment entry point: static rules only, no Qwen dependency at runtime."""

from neurosymbolic_fsad.inference import main


if __name__ == "__main__":
    main()

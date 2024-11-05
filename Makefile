# Version for production deployment
VERSION := 1.2

# Docker registry and project
REGISTRY := nvcr.io/pfteb4cqjzrs/playground

# List of services to build
SERVICES := api-service agent-service pdf-service tts-service

# Required environment variables
REQUIRED_ENV_VARS := ELEVENLABS_API_KEY NIM_KEY MAX_CONCURRENT_REQUESTS

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
NC := \033[0m  # No Color

# Explicitly use bash
SHELL := /bin/bash

# Check if environment variables are set
check_env:
	@for var in $(REQUIRED_ENV_VARS); do \
		if [ -z "$$(eval echo "\$$$$var")" ]; then \
			echo "$(RED)Error: $$var is not set$(NC)"; \
			echo "Please set required environment variables:"; \
			echo "  export $$var=<value>"; \
			exit 1; \
		else \
			echo "$(GREEN)✓ $$var is set$(NC)"; \
		fi \
	done

# Development target
dev: check_env
	@echo "$(GREEN)Starting development environment...$(NC)"
	docker compose -f docker-compose.yaml up --build

# Production target
prod: check_env
	@echo "$(GREEN)Starting production environment with version $(VERSION)...$(NC)"
	VERSION=$(VERSION) docker compose -f docker-compose.yaml up -d

# Version bump and release target
version-bump:
	@echo "Current version: $(VERSION)"
	@new_version=$$(echo $(VERSION) | awk -F. '{$$NF = $$NF + 1;} 1' | sed 's/ /./g'); \
	sed -i.bak "s/VERSION := 1.2
	rm Makefile.bak; \
	echo "$(GREEN)Version bumped to: $$new_version$(NC)"; \
	git add Makefile; \
	git commit -m "chore: bump version to $$new_version"; \
	git tag -a "v$$new_version" -m "Release v$$new_version"; \
	git push origin main; \
	git push origin "v$$new_version"

# Clean up containers and volumes
clean:
	docker compose -f docker-compose.yaml down -v

lint:
	ruff check

format: 
	ruff format

ruff: lint format

.PHONY: check_env dev clean ruff prod version-bump
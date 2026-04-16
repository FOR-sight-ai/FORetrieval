REGISTRY ?= ghcr.io
OWNER ?= random-plm
IMAGE_NAME ?= foretrieval-server
IMAGE_TAG ?= v0.0.1
IMAGE_REPO ?= $(REGISTRY)/$(OWNER)/$(IMAGE_NAME)
IMAGE ?= $(IMAGE_REPO):$(IMAGE_TAG)

# Backward-compatible aliases.
NAMESPACE ?= $(OWNER)
PROJECT ?= $(IMAGE_NAME)
DOCKERFILE ?= Dockerfile
BUILD_CONTEXT ?= .
PYTEST ?= pytest
TEST_ARGS ?= -m "not slow and not integration"
SRC_PACKAGE ?= foretrieval

.PHONY: check-buildx login-registry build publish run-server test-fast test-all coverage-fast coverage-all

check-buildx:
	@docker buildx version >/dev/null 2>&1 || { \
		echo "Error: docker buildx is not available."; \
		echo "Install Docker Buildx (or Docker Desktop), then run this command again."; \
		echo "Linux plugin package is often named 'docker-buildx-plugin'."; \
		exit 1; \
	}

login-registry:
	@test -n "$$GITHUB_PAT" || { \
		echo "Error: GITHUB_PAT is not set. Export a classic PAT with write:packages."; \
		exit 1; \
	}
	@printf '%s' "$$GITHUB_PAT" | docker login ghcr.io -u $(OWNER) --password-stdin

build: check-buildx
	docker buildx build --load -f $(DOCKERFILE) -t $(IMAGE) $(BUILD_CONTEXT)

publish: check-buildx login-registry
	docker buildx build --push -f $(DOCKERFILE) -t $(IMAGE) $(BUILD_CONTEXT)

run-server:
	REGISTRY=$(REGISTRY) OWNER=$(OWNER) IMAGE_NAME=$(IMAGE_NAME) IMAGE_TAG=$(IMAGE_TAG) ./scripts/run-docker.sh

test-fast:
	$(PYTEST) $(TEST_ARGS)

test-all:
	$(PYTEST)

coverage-fast:
	$(PYTEST) $(TEST_ARGS) --cov=$(SRC_PACKAGE) --cov-report=term-missing --cov-report=xml

coverage-all:
	$(PYTEST) --cov=$(SRC_PACKAGE) --cov-report=term-missing --cov-report=xml

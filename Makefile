REGISTRY ?= ghcr.io
OWNER ?= random-plm
IMAGE_NAME ?= foretrieval-server
IMAGE_TAG ?= v0.0.1
IMAGE_REPO ?= $(REGISTRY)/$(OWNER)/$(IMAGE_NAME)
IMAGE ?= $(IMAGE_REPO):$(IMAGE_TAG)
IMAGE_CPU ?= $(IMAGE_REPO):$(IMAGE_TAG)-base
GPU_CUDA_VERSION ?= 12.2.2
GPU_CUDNN_MAJOR ?= 8
HOST_ARCH_RAW := $(shell uname -m)
HOST_ARCH ?= $(if $(filter x86_64 amd64,$(HOST_ARCH_RAW)),amd64,$(if $(filter aarch64 arm64,$(HOST_ARCH_RAW)),arm64,$(HOST_ARCH_RAW)))
IMAGE_GPU ?= $(IMAGE_REPO):$(IMAGE_TAG)-cuda$(GPU_CUDA_VERSION)-cudnn$(GPU_CUDNN_MAJOR)-flashattn-bnb-$(HOST_ARCH)

# Backward-compatible aliases.
NAMESPACE ?= $(OWNER)
PROJECT ?= $(IMAGE_NAME)
DOCKERFILE ?= Dockerfile
BUILD_CONTEXT ?= .
PYTEST ?= pytest
TEST_ARGS ?= -m "not slow and not integration"
SRC_PACKAGE ?= foretrieval

.PHONY: check-buildx login-registry build build-cpu build-gpu publish publish-cpu publish-gpu run-server test-fast test-all coverage-fast coverage-all

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

build: build-cpu build-gpu

build-cpu: check-buildx
	docker buildx build --load -f $(DOCKERFILE) --target cpu -t $(IMAGE_CPU) $(BUILD_CONTEXT)

build-gpu: check-buildx
	docker buildx build --load -f $(DOCKERFILE) --target gpu -t $(IMAGE_GPU) $(BUILD_CONTEXT)

publish: publish-cpu publish-gpu

publish-cpu: check-buildx login-registry
	docker buildx build --push -f $(DOCKERFILE) --target cpu -t $(IMAGE_CPU) $(BUILD_CONTEXT)

publish-gpu: check-buildx login-registry
	docker buildx build --push -f $(DOCKERFILE) --target gpu -t $(IMAGE_GPU) $(BUILD_CONTEXT)

run-server:
	IMAGE=$(IMAGE_GPU) REGISTRY=$(REGISTRY) OWNER=$(OWNER) IMAGE_NAME=$(IMAGE_NAME) IMAGE_TAG=$(IMAGE_TAG) ./scripts/run-docker.sh

test-fast:
	$(PYTEST) $(TEST_ARGS)

test-all:
	$(PYTEST)

coverage-fast:
	$(PYTEST) $(TEST_ARGS) --cov=$(SRC_PACKAGE) --cov-report=term-missing --cov-report=xml

coverage-all:
	$(PYTEST) --cov=$(SRC_PACKAGE) --cov-report=term-missing --cov-report=xml

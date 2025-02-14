format:
	@cargo +nightly fmt

lint:
	@cargo +nightly fmt --check
	@cargo clippy -- -D warnings

asm:
	@cargo rustc -r -p shortcut -- --emit asm

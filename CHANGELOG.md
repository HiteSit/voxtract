# Changelog

All notable changes to Voxtract will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Session-based workflow: inbox → staging → finalize into named recording directories
- Speaker-grouped markdown output (## Speaker N headings)
- Multi-file support: multiple audio files merged into a single transcript with `# Part:` headers
- `clean_transcript` MCP prompt for post-processing via Claude
- Language selection with automatic timestamp handling (13 languages)
- Context biasing for domain-specific vocabulary
- Slug-based directory naming from transcript content

[Unreleased]: https://github.com/HiteSit/voxtract/compare/HEAD

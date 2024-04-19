import logging
import os
import time
from typing import Dict, Iterator, Optional, Tuple

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.utils.openai import is_openai_v1

logger = logging.getLogger(__name__)


class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files.

    Audio transcription is with OpenAI Whisper model.

    Args:
        api_key: OpenAI API key
        chunk_duration_threshold: minimum duration of a chunk in seconds
            NOTE: According to the OpenAI API, the chunk duration should be at least 0.1
            seconds. If the chunk duration is less or equal than the threshold,
            it will be skipped.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        chunk_duration_threshold: float = 0.1,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.chunk_duration_threshold = chunk_duration_threshold
        self.base_url = (
            base_url if base_url is not None else os.environ.get("OPENAI_API_BASE")
        )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io

        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not found, please install it with "
                "`pip install openai`"
            )
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub package not found, please install it with " "`pip install pydub`"
            )

        if is_openai_v1():
            # api_key optional, defaults to `os.environ['OPENAI_API_KEY']`
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            # Set the API key if provided
            if self.api_key:
                openai.api_key = self.api_key
            if self.base_url:
                openai.base_url = self.base_url

        # Audio file from disk
        audio = AudioSegment.from_file(blob.path)

        # Define the duration of each chunk in minutes
        # Need to meet 25MB size limit for Whisper API
        chunk_duration = 20
        chunk_duration_ms = chunk_duration * 60 * 1000

        # Split the audio into chunk_duration_ms chunks
        for split_number, i in enumerate(range(0, len(audio), chunk_duration_ms)):
            # Audio chunk
            chunk = audio[i : i + chunk_duration_ms]
            # Skip chunks that are too short to transcribe
            if chunk.duration_seconds <= self.chunk_duration_threshold:
                continue
            file_obj = io.BytesIO(chunk.export(format="mp3").read())
            if blob.source is not None:
                file_obj.name = blob.source + f"_part_{split_number}.mp3"
            else:
                file_obj.name = f"part_{split_number}.mp3"

            # Transcribe
            print(f"Transcribing part {split_number + 1}!")  # noqa: T201
            attempts = 0
            while attempts < 3:
                try:
                    if is_openai_v1():
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", file=file_obj
                        )
                    else:
                        transcript = openai.Audio.transcribe("whisper-1", file_obj)
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed. Exception: {str(e)}")  # noqa: T201
                    time.sleep(5)
            else:
                print("Failed to transcribe after 3 attempts.")  # noqa: T201
                continue

            yield Document(
                page_content=transcript.text,
                metadata={"source": blob.source, "chunk": split_number},
            )


class OpenAIWhisperParserLocal(BaseBlobParser):
    """Transcribe and parse audio files with OpenAI Whisper model.

    Audio transcription with OpenAI Whisper model locally from transformers.

    Parameters:
    device - device to use
        NOTE: By default uses the gpu if available,
        if you want to use cpu, please set device = "cpu"
    lang_model - whisper model to use, for example "openai/whisper-medium"
    forced_decoder_ids - id states for decoder in multilanguage model,
        usage example:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="french",
          task="transcribe")
        forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="french",
        task="translate")



    """

    def __init__(
        self,
        device: str = "0",
        lang_model: Optional[str] = None,
        batch_size: int = 8,
        chunk_length: int = 30,
        forced_decoder_ids: Optional[Tuple[Dict]] = None,
    ):
        """Initialize the parser.

        Args:
            device: device to use.
            lang_model: whisper model to use, for example "openai/whisper-medium".
              Defaults to None.
            forced_decoder_ids: id states for decoder in a multilanguage model.
              Defaults to None.
            batch_size: batch size used for decoding
              Defaults to 8.
            chunk_length: chunk length used during inference.
              Defaults to 30s.
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers package not found, please install it with "
                "`pip install transformers`"
            )
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch package not found, please install it with " "`pip install torch`"
            )

        # Determine the device to use
        if device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            default_model = "openai/whisper-base"
            self.lang_model = lang_model if lang_model else default_model
        else:
            # Set the language model based on the device and available memory
            mem = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)
            if mem < 5000:
                rec_model = "openai/whisper-base"
            elif mem < 7000:
                rec_model = "openai/whisper-small"
            elif mem < 12000:
                rec_model = "openai/whisper-medium"
            else:
                rec_model = "openai/whisper-large"
            self.lang_model = lang_model if lang_model else rec_model

        print("Using the following model: ", self.lang_model)  # noqa: T201

        self.batch_size = batch_size

        # load model for inference
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.lang_model,
            chunk_length_s=chunk_length,
            device=self.device,
        )
        if forced_decoder_ids is not None:
            try:
                self.pipe.model.config.forced_decoder_ids = forced_decoder_ids
            except Exception as exception_text:
                logger.info(
                    "Unable to set forced_decoder_ids parameter for whisper model"
                    f"Text of exception: {exception_text}"
                    "Therefore whisper model will use default mode for decoder"
                )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io

        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub package not found, please install it with `pip install pydub`"
            )

        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa package not found, please install it with "
                "`pip install librosa`"
            )

        # Audio file from disk
        audio = AudioSegment.from_file(blob.path)

        file_obj = io.BytesIO(audio.export(format="mp3").read())

        # Transcribe
        print(f"Transcribing part {blob.path}!")  # noqa: T201

        y, sr = librosa.load(file_obj, sr=16000)

        prediction = self.pipe(y.copy(), batch_size=self.batch_size)["text"]

        yield Document(
            page_content=prediction,
            metadata={"source": blob.source},
        )


class YandexSTTParser(BaseBlobParser):
    """Transcribe and parse audio files.
    Audio transcription is with OpenAI Whisper model."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        iam_token: Optional[str] = None,
        model: str = "general",
        language: str = "auto",
    ):
        """Initialize the parser.

        Args:
            api_key: API key for a service account
            with the `ai.speechkit-stt.user` role.
            iam_token: IAM token for a service account
            with the `ai.speechkit-stt.user` role.
            model: Recognition model name.
              Defaults to general.
            language: The language in ISO 639-1 format.
              Defaults to automatic language recognition.
        Either `api_key` or `iam_token` must be provided, but not both.
        """
        if (api_key is None) == (iam_token is None):
            raise ValueError(
                "Either 'api_key' or 'iam_token' must be provided, but not both."
            )
        self.api_key = api_key
        self.iam_token = iam_token
        self.model = model
        self.language = language

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        try:
            from speechkit import configure_credentials, creds, model_repository
            from speechkit.stt import AudioProcessingType
        except ImportError:
            raise ImportError(
                "yandex-speechkit package not found, please install it with "
                "`pip install yandex-speechkit`"
            )
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub package not found, please install it with " "`pip install pydub`"
            )

        if self.api_key:
            configure_credentials(
                yandex_credentials=creds.YandexCredentials(api_key=self.api_key)
            )
        else:
            configure_credentials(
                yandex_credentials=creds.YandexCredentials(iam_token=self.iam_token)
            )

        audio = AudioSegment.from_file(blob.path)

        model = model_repository.recognition_model()

        model.model = self.model
        model.language = self.language
        model.audio_processing_type = AudioProcessingType.Full

        result = model.transcribe(audio)

        for res in result:
            yield Document(
                page_content=res.normalized_text,
                metadata={"source": blob.source},
            )


class FasterWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files with faster-whisper.

    faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2,
    which is up to 4 times faster than openai/whisper for the same accuracy while using
    less memory. The efficiency can be further improved with 8-bit quantization on both
    CPU and GPU.

    It can automatically detect the following 14 languages and transcribe the text
    into their respective languages: en, zh, fr, de, ja, ko, ru, es, th, it, pt, vi,
    ar, tr.

    The gitbub repository for faster-whisper is :
    https://github.com/SYSTRAN/faster-whisper

    Example: Load a YouTube video and transcribe the video speech into a document.
        .. code-block:: python

            from langchain.document_loaders.generic import GenericLoader
            from langchain_community.document_loaders.parsers.audio
                import FasterWhisperParser
            from langchain.document_loaders.blob_loaders.youtube_audio
                import YoutubeAudioLoader


            url="https://www.youtube.com/watch?v=your_video"
            save_dir="your_dir/"
            loader = GenericLoader(
                YoutubeAudioLoader([url],save_dir),
                FasterWhisperParser()
            )
            docs = loader.load()

    """

    def __init__(
        self,
        *,
        device: Optional[str] = "cuda",
        model_size: Optional[str] = None,
    ):
        """Initialize the parser.

        Args:
            device: It can be "cuda" or "cpu" based on the available device.
            model_size: There are four model sizes to choose from: "base", "small",
                        "medium", and "large-v3", based on the available GPU memory.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch package not found, please install it with `pip install torch`"
            )

        # Determine the device to use
        if device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine the model_size
        if self.device == "cpu":
            self.model_size = "base"
        else:
            # Set the model_size based on the available memory
            mem = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)
            if mem < 1000:
                self.model_size = "base"
            elif mem < 3000:
                self.model_size = "small"
            elif mem < 5000:
                self.model_size = "medium"
            else:
                self.model_size = "large-v3"
        # If the user has assigned a model size, then use the assigned size
        if model_size is not None:
            if model_size in ["base", "small", "medium", "large-v3"]:
                self.model_size = model_size

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io

        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub package not found, please install it with `pip install pydub`"
            )

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster_whisper package not found, please install it with "
                "`pip install faster-whisper`"
            )

        # get the audio
        if isinstance(blob.data, bytes):
            # blob contains the audio
            audio = AudioSegment.from_file(io.BytesIO(blob.data))
        elif blob.data is None and blob.path:
            # Audio file from disk
            audio = AudioSegment.from_file(blob.path)
        else:
            raise ValueError("Unable to get audio from blob")

        file_obj = io.BytesIO(audio.export(format="mp3").read())

        # Transcribe
        model = WhisperModel(
            self.model_size, device=self.device, compute_type="float16"
        )

        segments, info = model.transcribe(file_obj, beam_size=5)

        for segment in segments:
            yield Document(
                page_content=segment.text,
                metadata={
                    "source": blob.source,
                    "timestamps": "[%.2fs -> %.2fs]" % (segment.start, segment.end),
                    "language": info.language,
                    "probability": "%d%%" % round(info.language_probability * 100),
                    **blob.metadata,
                },
            )

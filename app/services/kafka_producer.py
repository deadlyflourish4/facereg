"""
Kafka Producer for Face Events
Publishes face detection/recognition events to Kafka topics.
"""
import json
import logging
from datetime import datetime
from app.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_FACE_EVENTS, KAFKA_TOPIC_ALERTS, KAFKA_ENABLED

logger = logging.getLogger(__name__)

_producer = None


def _get_producer():
    """Lazy-init Kafka producer."""
    global _producer
    if _producer is None and KAFKA_ENABLED:
        try:
            from kafka import KafkaProducer
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
            logger.info(f"Kafka producer connected to {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.warning(f"Kafka unavailable: {e}. Events will be logged only.")
    return _producer


def publish_event(event_type, data, key=None):
    """Publish an event to the face-events topic.

    Args:
        event_type: Type of event (detection, register, verify, identify, spoof_alert)
        data: Event payload dict
        key: Optional partition key (e.g. user_id)
    """
    event = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }

    # Always log
    logger.info(f"[FACE_EVENT] {event_type}: {json.dumps(data, default=str)[:200]}")

    if not KAFKA_ENABLED:
        return

    producer = _get_producer()
    if producer is None:
        return

    try:
        topic = KAFKA_TOPIC_ALERTS if event_type == "spoof_alert" else KAFKA_TOPIC_FACE_EVENTS
        producer.send(topic, value=event, key=key)
        producer.flush()
    except Exception as e:
        logger.error(f"Failed to publish Kafka event: {e}")


def publish_detection_event(total_faces, processing_time):
    """Publish a face detection event."""
    publish_event("detection", {
        "total_faces": total_faces,
        "processing_time_ms": round(processing_time * 1000, 2),
    })


def publish_register_event(user_id, name, status, images_count):
    """Publish a user registration event."""
    publish_event("register", {
        "user_id": user_id,
        "name": name,
        "status": status,
        "images_processed": images_count,
    }, key=user_id)


def publish_verify_event(user_id, match, similarity):
    """Publish a verification event."""
    publish_event("verify", {
        "user_id": user_id,
        "match": match,
        "similarity": similarity,
    }, key=user_id)


def publish_identify_event(candidates_count, top_match=None):
    """Publish an identification event."""
    publish_event("identify", {
        "candidates_found": candidates_count,
        "top_match": top_match,
    })


def publish_spoof_alert(score, details):
    """Publish a spoofing alert."""
    publish_event("spoof_alert", {
        "score": score,
        "details": details,
    })

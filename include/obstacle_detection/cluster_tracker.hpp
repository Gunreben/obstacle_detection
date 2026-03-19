#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

namespace obstacle_detection
{

struct DetectionGeometry
{
  std::array<float, 3> center {0.0F, 0.0F, 0.0F};
  std::array<float, 3> size {0.0F, 0.0F, 0.0F};
  float range_xy {0.0F};
};

struct TrackedDetection
{
  int track_id {0};
  std::size_t detection_index {0};
  DetectionGeometry geometry;
  bool is_new_track {false};
};

class ClusterTracker
{
public:
  std::vector<TrackedDetection> update(
    const std::vector<DetectionGeometry> & detections,
    double base_gate,
    double range_gate_scale,
    int max_missed_frames,
    double smoothing_alpha)
  {
    smoothing_alpha = std::clamp(smoothing_alpha, 0.0, 1.0);
    max_missed_frames = std::max(0, max_missed_frames);

    struct CandidateMatch
    {
      double distance;
      std::size_t track_index;
      std::size_t detection_index;
    };

    std::vector<CandidateMatch> candidates;
    candidates.reserve(tracks_.size() * detections.size());

    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
      const auto & track = tracks_[track_index];
      for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
        const auto distance = distanceBetween(track.geometry, detections[detection_index]);
        const auto gate = base_gate + range_gate_scale * detections[detection_index].range_xy;
        if (distance <= gate) {
          candidates.push_back({distance, track_index, detection_index});
        }
      }
    }

    std::sort(
      candidates.begin(),
      candidates.end(),
      [](const CandidateMatch & lhs, const CandidateMatch & rhs) {
        return std::tie(lhs.distance, lhs.track_index, lhs.detection_index) <
               std::tie(rhs.distance, rhs.track_index, rhs.detection_index);
      });

    std::vector<bool> track_taken(tracks_.size(), false);
    std::vector<bool> detection_taken(detections.size(), false);
    std::vector<TrackedDetection> tracked_detections;
    tracked_detections.reserve(detections.size());

    for (const auto & candidate : candidates) {
      if (track_taken[candidate.track_index] || detection_taken[candidate.detection_index]) {
        continue;
      }

      auto & track = tracks_[candidate.track_index];
      track.geometry = blendGeometry(track.geometry, detections[candidate.detection_index], smoothing_alpha);
      track.missed_frames = 0;
      track_taken[candidate.track_index] = true;
      detection_taken[candidate.detection_index] = true;
      tracked_detections.push_back(
        {track.id, candidate.detection_index, track.geometry, false});
    }

    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
      if (!track_taken[track_index]) {
        tracks_[track_index].missed_frames += 1;
      }
    }

    tracks_.erase(
      std::remove_if(
        tracks_.begin(),
        tracks_.end(),
        [max_missed_frames](const TrackState & track) {
          return track.missed_frames > max_missed_frames;
        }),
      tracks_.end());

    for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
      if (detection_taken[detection_index]) {
        continue;
      }

      TrackState track;
      track.id = next_track_id_++;
      track.geometry = detections[detection_index];
      track.missed_frames = 0;
      tracks_.push_back(track);
      tracked_detections.push_back({track.id, detection_index, track.geometry, true});
    }

    std::sort(
      tracked_detections.begin(),
      tracked_detections.end(),
      [](const TrackedDetection & lhs, const TrackedDetection & rhs) {
        return lhs.detection_index < rhs.detection_index;
      });

    return tracked_detections;
  }

private:
  struct TrackState
  {
    int id {0};
    DetectionGeometry geometry;
    int missed_frames {0};
  };

  static double distanceBetween(
    const DetectionGeometry & lhs,
    const DetectionGeometry & rhs)
  {
    const auto dx = static_cast<double>(lhs.center[0] - rhs.center[0]);
    const auto dy = static_cast<double>(lhs.center[1] - rhs.center[1]);
    const auto dz = static_cast<double>(lhs.center[2] - rhs.center[2]);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  static DetectionGeometry blendGeometry(
    const DetectionGeometry & previous,
    const DetectionGeometry & current,
    double smoothing_alpha)
  {
    const auto prev_weight = static_cast<float>(1.0 - smoothing_alpha);
    const auto curr_weight = static_cast<float>(smoothing_alpha);

    DetectionGeometry blended;
    for (std::size_t i = 0; i < blended.center.size(); ++i) {
      blended.center[i] = previous.center[i] * prev_weight + current.center[i] * curr_weight;
      blended.size[i] = previous.size[i] * prev_weight + current.size[i] * curr_weight;
    }

    blended.range_xy = std::hypot(blended.center[0], blended.center[1]);
    return blended;
  }

  int next_track_id_ {0};
  std::vector<TrackState> tracks_;
};

}  // namespace obstacle_detection

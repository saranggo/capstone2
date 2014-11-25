function [ds, bs, trees] = imgdetect(im, model, thresh)
% Wrapper around gdetect.m that computes detections in an image.
%   [ds, bs, trees] = imgdetect(im, model, thresh)
%
% Return values (see gdetect.m)
%
% Arguments
%   im        Input image
%   model     Model to use for detection
%   thresh    Detection threshold (scores must be > thresh)

im = color(im);
pyra = featpyramid(im, model);

if model.features.extra_octave
    %TODO
    'TODO'
else
    pyra.valid_levels(3:10)=zeros(8,1);
    pyra.valid_levels(13:20)=zeros(8,1);
end
[ds, bs, trees] = gdetect(pyra, model, thresh);

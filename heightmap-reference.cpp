bool Spectrogram::Reference::operator==(const Spectrogram::Reference &b) const
{
    return log2_samples_size == b.log2_samples_size
            && block_index == b.block_index
            && _spectrogram == b._spectrogram;
}

void Spectrogram::Reference::getArea( Position &a, Position &b) const
{
    Position blockSize( _spectrogram->samples_per_block() * pow(2,log2_samples_size[0]),
                        _spectrogram->scales_per_block() * pow(2,log2_samples_size[1]));
    a.time = blockSize.time * block_index[0];
    a.scale = blockSize.scale * block_index[1];
    b.time = a.time + blockSize.time;
    b.scale = a.scale + blockSize.scale;
}

/* child references */
Spectrogram::Reference Spectrogram::Reference::left() {
    Reference r = *this;
    r.log2_samples_size[0]--;
    r.block_index[0]<<=1;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::right() {
    Reference r = *this;
    r.log2_samples_size[0]--;
    (r.block_index[0]<<=1)++;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::top() {
    Reference r = *this;
    r.log2_samples_size[1]--;
    r.block_index[1]<<=1;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::bottom() {
    Reference r = *this;
    r.log2_samples_size[1]--;
    (r.block_index[1]<<=1)++;
    return r;
}

/* sibblings, 3 other references who share the same parent */
Spectrogram::Reference Spectrogram::Reference::sibbling1() {
    Reference r = *this;
    r.block_index[0]^=1;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::sibbling2() {
    Reference r = *this;
    r.block_index[1]^=1;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::sibbling3() {
    Reference r = *this;
    r.block_index[0]^=1;
    r.block_index[1]^=1;
    return r;
}

/* parent */
Spectrogram::Reference Spectrogram::Reference::parent() {
    Reference r = *this;
    r.log2_samples_size[0]++;
    r.log2_samples_size[1]++;
    r.block_index[0]>>=1;
    r.block_index[1]>>=1;
    return r;
}

Spectrogram::Reference::Reference(Spectrogram *spectrogram)
:   _spectrogram(spectrogram)
{}

bool Spectrogram::Reference::containsSpectrogram() const
{
    Position a, b;
    getArea( a, b );

    if (b.time-a.time < _spectrogram->min_sample_size().time*_spectrogram->_samples_per_block )
        return false;
    //float msss = _spectrogram->min_sample_size().scale;
    //unsigned spb = _spectrogram->_scales_per_block;
    //float ms = msss*spb;
    if (b.scale-a.scale < _spectrogram->min_sample_size().scale*_spectrogram->_scales_per_block )
        return false;

    pTransform t = _spectrogram->transform();
    Signal::pSource wf = t->original_waveform();
    if (a.time >= wf->length() )
        return false;

    if (a.scale >= 1)
        return false;

    return true;
}

bool Spectrogram::Reference::toLarge() const
{
    Position a, b;
    getArea( a, b );
    pTransform t = _spectrogram->transform();
    Signal::pSource wf = t->original_waveform();
    if (b.time > 2 * wf->length() && b.scale > 2 )
        return true;
    return false;
}

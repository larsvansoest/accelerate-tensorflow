{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE UndecidableSuperClasses #-}

module Data.Accelerate.TensorFlow.Type where

import Data.Array.Accelerate.Representation.Shape hiding (union)
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.Array.Unique (UniqueArray(UniqueArray), withUniqueArrayPtr)
import Data.Array.Accelerate.Lifetime
import GHC.ForeignPtr
import Control.Monad.IO.Class (liftIO)
import Data.Array.Accelerate.Type (ScalarType (..), SingleType (NumSingleType), NumType (IntegralNumType, FloatingNumType), IntegralType (..), FloatingType (..), VectorType (..))

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import qualified TensorFlow.Internal.FFI as FFI
import qualified Data.Vector as V
import Data.Int
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Foreign (Ptr, castPtr, Word8)
import Unsafe.Coerce (unsafeCoerce)
import Data.Array.Accelerate.Analysis.Match ( type (:~:)(Refl), matchScalarType )
import qualified Data.Functor.Identity as TF
import qualified Data.Vector.Storable as S
import Text.Printf
import qualified Data.Complex
import qualified Data.Word
import qualified Data.ByteString
import Data.Array.Accelerate.Representation.Type (TypeR)

import Data.Type.Equality ((:~:)(..))
import Data.Proxy (Proxy(..))
import Data.Kind (Type)
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}

import GHC.TypeLits
import Data.Kind
import Data.Proxy
import Data.Type.Bool (If)

type TFAdd = '[Data.Complex.Complex Double,
               Data.Complex.Complex Float,
               Data.ByteString.ByteString, 
               Data.Int.Int16,
               Data.Int.Int32, 
               Data.Int.Int64,
               Data.Int.Int8, 
               Data.Word.Word16,
               Data.Word.Word8, 
               Double, 
               Float]

data TensorDict types t where
  TensorDict :: TF.OneOf types t => TensorDict types t

data TensorTypeDict a where
  TensorTypeDict :: (S.Storable a, TF.TensorDataType S.Vector a, TF.TensorType a) => TensorTypeDict a

tensorTypeDict :: ScalarType a -> TensorTypeDict a
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt)))    = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt8)))   = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt16)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt32)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeInt64)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord)))   = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord8)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord16))) = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord32))) = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType TypeWord64))) = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeHalf)))   = error "not a tensortype"
tensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeFloat)))  = TensorTypeDict
tensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType TypeDouble))) = TensorTypeDict
tensorTypeDict (VectorScalarType _)                                            = error "not a tensortype"